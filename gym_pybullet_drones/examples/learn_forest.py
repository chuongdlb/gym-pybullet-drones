"""MAVSAC training: 3 drones sharing vision to navigate a dense forest.

Uses the novel Multi-Agent Vision-Sharing Actor-Critic (MAVSAC) algorithm
with CNN vision encoding + cross-attention communication between agents,
trained via PPO.

Example
-------
    $ uv run python gym_pybullet_drones/examples/learn_forest.py
    $ uv run python gym_pybullet_drones/examples/learn_forest.py --timesteps 5000000
    $ uv run python gym_pybullet_drones/examples/learn_forest.py --play results/forest-*/best_model.zip

"""
import os
import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_pybullet_drones.envs.DenseForestAviary import DenseForestAviary
from gym_pybullet_drones.algorithms.mavsac import MAVSACPolicy, CrossAttentionVisionExtractor
from gym_pybullet_drones.utils.enums import ActionType
from gym_pybullet_drones.utils.utils import sync

# Defaults
DEFAULT_ACT = ActionType('rpm')
DEFAULT_TIMESTEPS = 5_000_000
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_ENVS = 2
DEFAULT_NUM_AGENTS = 3


class ForestProgressCallback(BaseCallback):
    """Log cooperative navigation metrics during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_waypoints = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "waypoints_reached" in info:
                self.episode_waypoints.append(info["waypoints_reached"])

        if self.n_calls % 5000 == 0 and len(self.episode_waypoints) > 0:
            recent = self.episode_waypoints[-100:]
            avg_wp = np.mean(recent)
            max_wp = np.max(recent)
            self.logger.record("forest/avg_waypoints_reached", avg_wp)
            self.logger.record("forest/max_waypoints_reached", max_wp)
            if self.verbose > 0:
                print(f"[Forest] Avg WP: {avg_wp:.1f}, Max WP: {max_wp}")
        return True


def make_env(gui=False, record=False):
    """Create a DenseForestAviary environment."""
    return DenseForestAviary(
        num_drones=DEFAULT_NUM_AGENTS,
        act=DEFAULT_ACT,
        gui=gui,
        record=record,
    )


def get_policy_kwargs():
    """MAVSAC policy kwargs for PPO."""
    return dict(
        features_extractor_class=CrossAttentionVisionExtractor,
        features_extractor_kwargs=dict(
            num_agents=DEFAULT_NUM_AGENTS,
            embed_dim=64,
            num_heads=4,
            features_dim=128,
            vision_out_dim=48,
            state_out_dim=16,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )


def train(output_folder=DEFAULT_OUTPUT_FOLDER,
          total_timesteps=DEFAULT_TIMESTEPS,
          n_envs=DEFAULT_NUM_ENVS):
    """Train MAVSAC policy on DenseForestAviary."""

    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    filename = os.path.join(output_folder, f'forest-{timestamp}')
    os.makedirs(filename, exist_ok=True)

    print("=" * 60)
    print("MAVSAC: Multi-Agent Vision-Sharing Actor-Critic")
    print("=" * 60)
    print(f"  Agents:          {DEFAULT_NUM_AGENTS}")
    print(f"  Observation:     RGB (shared vision) + kinematic state")
    print(f"  Action type:     {DEFAULT_ACT}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs:   {n_envs}")
    print(f"  Output:          {filename}")
    print("=" * 60)

    # Create training environments
    train_env = make_vec_env(
        DenseForestAviary,
        env_kwargs=dict(num_drones=DEFAULT_NUM_AGENTS, act=DEFAULT_ACT),
        n_envs=n_envs,
        seed=0,
        vec_env_cls=DummyVecEnv,
    )

    eval_env = DenseForestAviary(num_drones=DEFAULT_NUM_AGENTS, act=DEFAULT_ACT)

    print(f"[INFO] Action space: {train_env.action_space}")
    print(f"[INFO] Observation space: {train_env.observation_space}")

    # Build MAVSAC PPO model
    policy_kwargs = get_policy_kwargs()

    model = PPO(
        MAVSACPolicy,
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(filename, 'tb'),
        verbose=1,
    )

    # Architecture summary
    print("\n[MAVSAC Architecture]")
    print(f"  Vision encoder: SharedCNN → {policy_kwargs['features_extractor_kwargs']['vision_out_dim']}D per drone")
    print(f"  State encoder:  MLP → {policy_kwargs['features_extractor_kwargs']['state_out_dim']}D per drone")
    print(f"  Cross-attention: {policy_kwargs['features_extractor_kwargs']['num_heads']} heads, dim={policy_kwargs['features_extractor_kwargs']['embed_dim']}")
    print(f"  Gated fusion:   sigmoid gate blending local + communicated")
    print(f"  Output:          {policy_kwargs['features_extractor_kwargs']['features_dim']}D")
    print(f"  Actor:           {policy_kwargs['net_arch']['pi']}")
    print(f"  Critic:          {policy_kwargs['net_arch']['vf']}")
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Total params:    {total_params:,}\n")

    # Callbacks
    target_reward = 400.0
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        verbose=1,
        best_model_save_path=filename,
        log_path=filename,
        eval_freq=2000,
        deterministic=True,
        render=False,
    )
    progress_callback = ForestProgressCallback(verbose=1)
    callbacks = CallbackList([eval_callback, progress_callback])

    # Train
    print("[INFO] Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=50,
    )

    model.save(os.path.join(filename, 'final_model'))
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
        plt.title('MAVSAC Training: Dense Forest Navigation (3 Drones, Shared Vision)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(filename, 'training_curve.png'), dpi=150)
        print(f"[INFO] Training curve saved to {filename}/training_curve.png")

    train_env.close()
    eval_env.close()
    return filename


def evaluate(model_path, gui=True, n_episodes=5):
    """Evaluate a trained MAVSAC model with visualization."""

    print(f"\n[INFO] Loading model from {model_path}")
    model = PPO.load(model_path)

    # Quantitative evaluation
    test_env_nogui = make_env(gui=False)
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print(f"\n[RESULT] Mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")
    test_env_nogui.close()

    if not gui:
        return

    # Visual evaluation
    test_env = make_env(gui=True)

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
                wp = info.get("waypoints_reached", 0)
                total = info.get("total_waypoints", 5)
                min_obs = info.get("min_obstacle_dist", float('inf'))
                print(f"  Step {step:4d} | WP: {wp}/{total} | "
                      f"Min obs dist: {min_obs:.2f}m | Reward: {total_reward:.1f}")

            if terminated or truncated:
                wp = info.get("waypoints_reached", 0)
                total = info.get("total_waypoints", 5)
                status = "COMPLETED" if info.get("all_waypoints_reached") else "TRUNCATED"
                print(f"  >> {status} | Waypoints: {wp}/{total} | Total reward: {total_reward:.1f}")
                break

    test_env.close()


def main():
    parser = argparse.ArgumentParser(
        description='MAVSAC: Multi-Agent Vision-Sharing Actor-Critic for Dense Forest Navigation'
    )
    parser.add_argument('--play', type=str, default=None,
                        help='Path to trained model for evaluation (skip training)')
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

    args = parser.parse_args()
    gui = args.gui.lower() in ('true', '1', 'yes')

    if args.play:
        evaluate(args.play, gui=gui, n_episodes=args.episodes)
    else:
        filename = train(
            output_folder=args.output_folder,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
        )

        best_model = os.path.join(filename, 'best_model.zip')
        if os.path.isfile(best_model):
            evaluate(best_model, gui=gui, n_episodes=3)
        else:
            final_model = os.path.join(filename, 'final_model.zip')
            if os.path.isfile(final_model):
                evaluate(final_model, gui=gui, n_episodes=3)


if __name__ == '__main__':
    main()
