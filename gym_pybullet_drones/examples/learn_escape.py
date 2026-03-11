"""Forest Escape v10b: Single-Drone + Higher Resolution.

v10b: prove single-drone navigation with 128×96 RGBD, then scale to multi-agent.
Supports --num_agents 1 (default) and --img_res 128,96 (default).

Example
-------
    $ python gym_pybullet_drones/examples/learn_escape.py --num_agents 1 --n_envs 6
    $ python gym_pybullet_drones/examples/learn_escape.py --num_agents 3 --img_res 64,48
    $ python gym_pybullet_drones/examples/learn_escape.py --play results/escape-*/best_model.zip
"""
import os
import time
import argparse
from datetime import datetime

import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_pybullet_drones.envs.ForestEscapeAviary import ForestEscapeAviary
from gym_pybullet_drones.utils.enums import ActionType
from gym_pybullet_drones.utils.utils import sync

# Defaults
DEFAULT_ACT = ActionType('pid')
DEFAULT_TIMESTEPS = 5_000_000
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_ENVS = 6  # 6 works for 1-drone 128×96; 8 for 3-drone 64×48
DEFAULT_NUM_AGENTS = 1
DEFAULT_IMG_RES = (128, 96)


class VisionStateExtractor(BaseFeaturesExtractor):
    """Shared CNN + optional cross-attention for RGBD vision.

    For n_drones > 1: cross-attention lets drones share visual information.
    For n_drones == 1: skip cross-attention (direct CNN → fusion).

    Single-drone (v10b):
        RGBD(96,128,4) → CNN(4→32→64→64) → Flatten → FC → 64D
        State(29D) → FC → 64D
        Concat(128D) → FC → 256D

    Multi-drone (v10):
        Per drone: RGBD → SharedCNN → FC → 64D token
        Cross-attention: 2-head self-attention over N tokens + residual + LayerNorm
        Pool: mean(N enhanced tokens) → 64D
        State + Fusion same as above.
    """

    def __init__(self, observation_space, features_dim=256,
                 n_drones=1, img_h=96, img_w=128, img_c=4,
                 token_dim=64, n_attn_heads=2):
        super().__init__(observation_space, features_dim)
        self.n_drones = n_drones
        self.token_dim = token_dim
        self.use_cross_attn = n_drones > 1

        # Shared CNN for per-drone images (NatureCNN-style, smaller)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output dim dynamically (adapts to any resolution)
        with torch.no_grad():
            sample = torch.zeros(1, img_c, img_h, img_w)
            cnn_out_dim = self.cnn(sample).shape[1]

        # Project CNN output to token_dim per drone
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, token_dim),
            nn.ReLU(),
        )

        # Cross-attention only for multi-drone
        if self.use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=n_attn_heads,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(token_dim)

        # State MLP
        state_dim = observation_space["state"].shape[0]
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )

        # Fusion: pooled vision (64D) + state (64D) → features_dim
        self.fusion = nn.Sequential(
            nn.Linear(token_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        vision = observations["vision"]   # (batch, N, H, W, 4)
        state = observations["state"]     # (batch, N*29)

        batch_size = vision.shape[0]

        # --- Per-drone CNN → tokens ---
        imgs = vision.reshape(batch_size * self.n_drones,
                              vision.shape[2], vision.shape[3], vision.shape[4])
        imgs = imgs.permute(0, 3, 1, 2).float() / 255.0

        cnn_out = self.cnn(imgs)                                # (batch*N, cnn_dim)
        tokens = self.cnn_proj(cnn_out)                         # (batch*N, token_dim)
        tokens = tokens.reshape(batch_size, self.n_drones, self.token_dim)

        # --- Cross-attention (multi-drone only) ---
        if self.use_cross_attn:
            attn_out, _ = self.cross_attn(tokens, tokens, tokens)
            tokens = self.attn_norm(tokens + attn_out)

        # --- Mean pool across drones (identity for n_drones=1) ---
        vision_embed = tokens.mean(dim=1)                      # (batch, 64)

        # --- State embedding ---
        state_embed = self.state_fc(state)                     # (batch, 64)

        # --- Fusion ---
        fused = torch.cat([vision_embed, state_embed], dim=1)  # (batch, 128)
        return self.fusion(fused)                              # (batch, 256)


class EscapeProgressCallback(BaseCallback):
    """Log escape progress metrics during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_x_progress = []
        self.episode_escapes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "x_progress" in info:
                self.episode_x_progress.append(info["x_progress"])
                self.episode_escapes.append(1.0 if info.get("all_escaped") else 0.0)

        if self.n_calls % 5000 == 0 and len(self.episode_x_progress) > 0:
            recent_progress = self.episode_x_progress[-100:]
            recent_escapes = self.episode_escapes[-100:]
            avg_progress = np.mean(recent_progress)
            escape_rate = np.mean(recent_escapes)
            self.logger.record("escape/avg_x_progress", avg_progress)
            self.logger.record("escape/escape_rate", escape_rate)
            if self.verbose > 0:
                print(f"[Escape] X-progress: {avg_progress:.2f}, Escape rate: {escape_rate:.1%}")
        return True


def make_env_fn(gui=False, record=False, num_agents=DEFAULT_NUM_AGENTS,
                img_res=DEFAULT_IMG_RES):
    """Return a callable that creates a ForestEscapeAviary (Dict obs, vision+state)."""
    def _init():
        return ForestEscapeAviary(
            num_drones=num_agents,
            act=DEFAULT_ACT,
            gui=gui,
            record=record,
            img_res=img_res,
        )
    return _init


def linear_schedule(initial_value: float, final_value: float = 1e-5):
    """Linear LR decay from initial_value to final_value over training."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def train(output_folder=DEFAULT_OUTPUT_FOLDER,
          total_timesteps=DEFAULT_TIMESTEPS,
          n_envs=DEFAULT_NUM_ENVS,
          resume_path=None,
          lr=3e-5, lr_final=5e-6,
          ent_coef=0.005, vf_coef=1.0,
          max_grad_norm=0.5,
          version_tag=None,
          num_agents=DEFAULT_NUM_AGENTS,
          img_res=DEFAULT_IMG_RES):
    """Train PPO with RGBD vision+state on ForestEscapeAviary."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_w, img_h = img_res
    state_dim = num_agents * ForestEscapeAviary.STATE_OBS_DIM  # 29 per drone

    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    tag = f'-{version_tag}' if version_tag else ''
    filename = os.path.join(output_folder, f'escape{tag}-{timestamp}')
    os.makedirs(filename, exist_ok=True)

    print("=" * 60)
    print(f"Forest Escape v10b: {num_agents}-drone, {img_w}×{img_h} RGBD")
    if resume_path:
        print(f"  RESUMING FROM: {resume_path}")
    print("=" * 60)
    print(f"  Device:          {device}")
    print(f"  Agents:          {num_agents}")
    print(f"  Image res:       {img_w}×{img_h}")
    print(f"  Observation:     Dict(vision=({num_agents},{img_h},{img_w},4) RGBD + state={state_dim}D)")
    print(f"  Action type:     {DEFAULT_ACT}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs:   {n_envs}")
    print(f"  LR:              {lr} → {lr_final}")
    print(f"  ent_coef:        {ent_coef}")
    print(f"  vf_coef:         {vf_coef}")
    print(f"  max_grad_norm:   {max_grad_norm}")
    print(f"  Output:          {filename}")
    print("=" * 60)

    # SubprocVecEnv for parallel physics (CPU), model on GPU
    env_kwargs = dict(num_agents=num_agents, img_res=img_res)
    train_env = SubprocVecEnv([make_env_fn(**env_kwargs) for _ in range(n_envs)])
    # norm_obs=False: vision is uint8 (CNN normalizes internally), state scales are reasonable
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Eval env
    eval_env_raw = DummyVecEnv([make_env_fn(**env_kwargs)])
    eval_env = VecNormalize(eval_env_raw, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, training=False)

    # Load VecNormalize stats from previous run if resuming
    if resume_path:
        vec_norm_path = os.path.join(os.path.dirname(resume_path), 'vec_normalize.pkl')
        if os.path.isfile(vec_norm_path):
            print(f"[INFO] Loading VecNormalize stats from {vec_norm_path}")
            train_env = VecNormalize.load(vec_norm_path, train_env.venv)
            eval_env = VecNormalize.load(vec_norm_path, eval_env_raw)
            eval_env.training = False

    print(f"[INFO] Action space: {train_env.action_space}")
    print(f"[INFO] Observation space: {train_env.observation_space}")

    policy_kwargs = dict(
        features_extractor_class=VisionStateExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            n_drones=num_agents,
            img_h=img_h,
            img_w=img_w,
        ),
        net_arch=dict(pi=[256], vf=[256]),  # smaller heads since extractor is 256D
        log_std_init=-0.7,
    )

    if resume_path:
        print(f"[INFO] Loading model from {resume_path}")
        model = PPO.load(
            resume_path,
            env=train_env,
            device=device,
            learning_rate=linear_schedule(lr, lr_final),
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=os.path.join(filename, 'tb'),
        )
    else:
        model = PPO(
            'MultiInputPolicy',
            train_env,
            policy_kwargs=policy_kwargs,
            device=device,
            learning_rate=linear_schedule(lr, lr_final),
            n_steps=2048,
            batch_size=128,
            n_epochs=3,  # fewer epochs to reduce KL divergence
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=os.path.join(filename, 'tb'),
            verbose=1,
        )

    # Architecture summary
    print(f"\n[v10b Architecture — {num_agents} drone(s), {img_w}×{img_h}]")
    print(f"  Policy:    MultiInputPolicy + VisionStateExtractor")
    print(f"  Vision:    SharedCNN(RGBD 4ch→32→64→64) per drone → 64D token")
    if num_agents > 1:
        print(f"  CrossAttn: 2-head self-attention over {num_agents} drone tokens + residual + LN")
        print(f"  Pool:      mean({num_agents} enhanced tokens) → 64D vision")
    else:
        print(f"  CrossAttn: SKIPPED (single drone)")
        print(f"  Pool:      squeeze → 64D vision")
    print(f"  State:     {state_dim}D → 64D")
    print(f"  Fusion:    128D → 256D → actor(256→{num_agents*3}) + critic(256→1)")
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Total params: {total_params:,}")
    print()

    # Callbacks
    target_reward = 500.0
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward, verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        verbose=1,
        best_model_save_path=filename,
        log_path=filename,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    progress_callback = EscapeProgressCallback(verbose=1)
    callbacks = CallbackList([eval_callback, progress_callback])

    # Train
    print("[INFO] Starting training...")
    if device == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=50,
    )

    model.save(os.path.join(filename, 'final_model'))
    train_env.save(os.path.join(filename, 'vec_normalize.pkl'))
    print(f"\n[INFO] Final model saved to {filename}/final_model.zip")

    # Plot training curve
    eval_file = os.path.join(filename, 'evaluations.npz')
    if os.path.isfile(eval_file):
        with np.load(eval_file) as data:
            timesteps = data['timesteps']
            results = data['results'][:, 0]

        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, results, marker='o', markersize=2, alpha=0.7)
        plt.xlabel('Training Steps')
        plt.ylabel('Episode Reward')
        plt.title(f'Forest Escape v10b ({num_agents} drone(s), {img_w}×{img_h})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(filename, 'training_curve.png'), dpi=150)
        print(f"[INFO] Training curve saved to {filename}/training_curve.png")

    train_env.close()
    eval_env.close()
    return filename


def evaluate(model_path, gui=True, n_episodes=5,
             num_agents=DEFAULT_NUM_AGENTS, img_res=DEFAULT_IMG_RES):
    """Evaluate a trained model."""
    print(f"\n[INFO] Loading model from {model_path}")
    model = PPO.load(model_path)

    # Quantitative evaluation
    test_env_nogui = ForestEscapeAviary(
        num_drones=num_agents, act=DEFAULT_ACT, gui=False, img_res=img_res,
    )
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print(f"\n[RESULT] Mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")
    test_env_nogui.close()

    if not gui:
        return

    # Visual evaluation
    test_env = ForestEscapeAviary(
        num_drones=num_agents, act=DEFAULT_ACT, gui=True, img_res=img_res,
    )

    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1}/{n_episodes} ---")
        obs, info = test_env.reset(seed=ep)
        start = time.time()
        total_reward = 0

        for step in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            test_env.render()
            sync(step, start, test_env.CTRL_TIMESTEP)

            if step % 30 == 0:
                escaped = info.get("num_escaped", 0)
                progress = info.get("x_progress", 0.0)
                mx = info.get("min_x", 0.0)
                print(f"  Step {step:4d} | Escaped: {escaped}/{num_agents} | "
                      f"X-progress: {progress:.1%} | Min X: {mx:.2f} | "
                      f"Reward: {total_reward:.1f}")

            if terminated or truncated:
                status = "ESCAPED" if info.get("all_escaped") else "TRUNCATED"
                progress = info.get("x_progress", 0.0)
                print(f"  >> {status} | X-progress: {progress:.1%} | "
                      f"Total reward: {total_reward:.1f}")
                break

    test_env.close()


def main():
    parser = argparse.ArgumentParser(
        description='Forest Escape v10b: Single-Drone + Higher Resolution',
    )
    parser.add_argument('--play', type=str, default=None,
                        help='Path to trained model for evaluation')
    parser.add_argument('--gui', type=str, default='true',
                        help='Enable GUI for evaluation (default: true)')
    parser.add_argument('--timesteps', type=int, default=DEFAULT_TIMESTEPS,
                        help=f'Total training timesteps (default: {DEFAULT_TIMESTEPS:,})')
    parser.add_argument('--n_envs', type=int, default=DEFAULT_NUM_ENVS,
                        help=f'Number of parallel environments (default: {DEFAULT_NUM_ENVS})')
    parser.add_argument('--output_folder', type=str, default=DEFAULT_OUTPUT_FOLDER,
                        help=f'Output folder (default: {DEFAULT_OUTPUT_FOLDER})')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes (default: 5)')
    # v10b: drone count and image resolution
    parser.add_argument('--num_agents', type=int, default=DEFAULT_NUM_AGENTS,
                        help=f'Number of drones (default: {DEFAULT_NUM_AGENTS})')
    parser.add_argument('--img_res', type=str, default=f'{DEFAULT_IMG_RES[0]},{DEFAULT_IMG_RES[1]}',
                        help=f'Image resolution W,H (default: {DEFAULT_IMG_RES[0]},{DEFAULT_IMG_RES[1]})')
    # Resume from checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model .zip to resume training from')
    # Tunable hyperparams (for auto-recovery)
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Initial learning rate (default: 3e-5)')
    parser.add_argument('--lr_final', type=float, default=5e-6,
                        help='Final learning rate (default: 5e-6)')
    parser.add_argument('--ent_coef', type=float, default=0.005,
                        help='Entropy coefficient (default: 0.005)')
    parser.add_argument('--vf_coef', type=float, default=1.0,
                        help='Value function coefficient (default: 1.0)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Max gradient norm (default: 0.5)')
    parser.add_argument('--version_tag', type=str, default=None,
                        help='Version tag for output folder (e.g. v10b-1drone)')

    args = parser.parse_args()
    gui = args.gui.lower() in ('true', '1', 'yes')

    # Parse image resolution "W,H"
    img_w, img_h = [int(x) for x in args.img_res.split(',')]
    img_res = (img_w, img_h)

    if args.play:
        evaluate(args.play, gui=gui, n_episodes=args.episodes,
                 num_agents=args.num_agents, img_res=img_res)
    else:
        filename = train(
            output_folder=args.output_folder,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            resume_path=args.resume,
            lr=args.lr,
            lr_final=args.lr_final,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            version_tag=args.version_tag,
            num_agents=args.num_agents,
            img_res=img_res,
        )

        best_model = os.path.join(filename, 'best_model.zip')
        if os.path.isfile(best_model):
            evaluate(best_model, gui=gui, n_episodes=3,
                     num_agents=args.num_agents, img_res=img_res)
        else:
            final_model = os.path.join(filename, 'final_model.zip')
            if os.path.isfile(final_model):
                evaluate(final_model, gui=gui, n_episodes=3,
                         num_agents=args.num_agents, img_res=img_res)


if __name__ == '__main__':
    main()
