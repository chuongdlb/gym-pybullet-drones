"""Optimized training script with parallel environments and CPU acceleration."""
import os
import time
from datetime import datetime
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import multiprocessing as mp

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_ENVS = 4
DEFAULT_TIMESTEPS = int(5e5)  # 500K steps

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

def make_env(env_id, gui=False, obs_type=DEFAULT_OBS, act_type=DEFAULT_ACT, seed=None):
    """Create environment factory function for SubprocVecEnv."""
    def _init():
        env = GateAviary(gui=gui, obs=obs_type, act=act_type)
        # GateAviary doesn't have seed method, so set numpy seed instead
        if seed is not None:
            np.random.seed(seed + env_id)
        return env
    return _init

def run(output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, 
        num_envs=DEFAULT_NUM_ENVS,
        timesteps=DEFAULT_TIMESTEPS,
        use_subprocess=True):
    """Train a drone to fly through 6 gates with parallel environments."""
    
    # Set multiprocessing start method for Windows
    mp.set_start_method('spawn', force=True)
    
    filename = os.path.join(output_folder, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    print('[INFO] Creating training environments')
    print(f'[INFO] Number of parallel environments: {num_envs}')
    
    # Create environment factory functions
    env_fns = [make_env(i, gui=gui, obs_type=DEFAULT_OBS, act_type=DEFAULT_ACT, seed=0) 
               for i in range(num_envs)]
    
    # Use SubprocVecEnv for parallel training (faster data collection)
    # Falls back to DummyVecEnv if subprocess fails
    try:
        if use_subprocess and num_envs > 1:
            print('[INFO] Creating SubprocVecEnv (parallel training)')
            train_env = SubprocVecEnv(env_fns)
        else:
            print('[INFO] Creating DummyVecEnv (single-threaded fallback)')
            train_env = DummyVecEnv(env_fns)
    except Exception as e:
        print(f'[WARNING] SubprocVecEnv failed: {e}')
        print('[INFO] Falling back to DummyVecEnv')
        train_env = DummyVecEnv(env_fns)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    print('[INFO] Creating PPO model with optimized hyperparameters')
    
    # Optimized PPO hyperparameters for faster convergence
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        n_steps=256,              # Larger batch size (was 128)
        batch_size=64,            # Minibatch size
        n_epochs=10,              # More epochs per update
        learning_rate=3e-4,       # Learning rate
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE lambda
        ent_coef=0.0,             # Entropy coefficient
        policy_kwargs=dict(
            net_arch=[256, 256]   # Larger network
        )
    )

    total_timesteps = int(timesteps)
    
    print(f'[INFO] Starting training for {total_timesteps:,} timesteps')
    print(f'[INFO] Models will be saved to: {filename}/')
    
    # Save checkpoint every 50K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // max(num_envs, 1), 1),
        save_path=filename,
        name_prefix='checkpoint',
        save_replay_buffer=False
    )
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        callback=checkpoint_callback
    )
    
    elapsed = time.time() - start_time
    print(f'[INFO] Training completed successfully in {elapsed:.1f}s')
    print(f'[INFO] Training speed: {total_timesteps/elapsed:.0f} steps/sec')

    #### Save the final model ########################################
    final_model_path = filename+'/final_model.zip'
    model.save(final_model_path)
    print(f'[INFO] Final model saved to: {final_model_path}')
    
    print(filename)
    train_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized gate flying training')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder for logs', metavar='')
    parser.add_argument('--num_envs',           default=DEFAULT_NUM_ENVS,      type=int,           help='Number of parallel environments', metavar='')
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int,           help='Total training timesteps', metavar='')
    parser.add_argument('--use_subprocess',     default=True,                  type=str2bool,      help='Use SubprocVecEnv (faster but may fail on Windows)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
