"""Optimized single-environment training with CPU and batch optimizations."""
import os
import time
from datetime import datetime
import argparse
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_TIMESTEPS = int(5e5)  # 500K steps

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

def run(output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, 
        timesteps=DEFAULT_TIMESTEPS):
    """Train a drone to fly through 6 gates with CPU optimizations."""
    
    filename = os.path.join(output_folder, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    print('[INFO] Creating training environment')
    train_env = GateAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    print('[INFO] Creating PPO model with CPU-optimized hyperparameters')
    
    # Optimized PPO hyperparameters for CPU training
    # - Larger batch size (256) = better gradient estimates, fewer updates needed
    # - More epochs (15) = squeeze more value from each batch
    # - Larger network (512, 512) = better function approximation
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        n_steps=256,              # Collect 256 steps per update (2x the default 128)
        batch_size=64,            # Minibatch size for SGD
        n_epochs=15,              # More gradient updates per batch (was 10)
        learning_rate=3e-4,       # Slightly higher learning rate
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE lambda
        clip_range=0.2,           # PPO clip range
        max_grad_norm=0.5,        # Gradient clipping
        vf_coef=0.5,              # Value function coefficient
        ent_coef=0.001,           # Small entropy bonus for exploration
        policy_kwargs=dict(
            net_arch=[512, 512],  # Larger network for better learning
            ortho_init=True        # Orthogonal weight initialization
        )
    )

    total_timesteps = int(timesteps)
    
    print(f'[INFO] Starting training for {total_timesteps:,} timesteps')
    print(f'[INFO] Models will be saved to: {filename}/')
    print(f'[INFO] CPU threading: OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS", "auto")}')
    
    start_time = time.time()
    fps_start = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=100
        )
        
        elapsed = time.time() - start_time
        print(f'\n[INFO] Training completed successfully in {elapsed:.1f}s')
        print(f'[INFO] Training speed: {total_timesteps/elapsed:.0f} steps/sec ({60*total_timesteps/elapsed:.0f} steps/min)')

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f'\n[INFO] Training interrupted after {elapsed:.1f}s')
        print(f'[INFO] Completed {model.num_timesteps:,} timesteps')

    #### Save the final model ########################################
    final_model_path = filename+'/final_model.zip'
    model.save(final_model_path)
    print(f'[INFO] Final model saved to: {final_model_path}')
    
    print(filename)
    train_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPU-optimized gate flying training')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder for logs', metavar='')
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int,           help='Total training timesteps', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
