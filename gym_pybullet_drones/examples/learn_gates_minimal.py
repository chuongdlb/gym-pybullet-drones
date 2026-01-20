"""Minimal training script for gate flying - direct training loop.

This script trains a drone with direct training loop to avoid PPO/DummyVecEnv issues.
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, local=True):
    """Train a drone to fly through 6 gates with minimal overhead."""
    
    filename = os.path.join(output_folder, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    print('[INFO] Creating training environment')
    train_env = GateAviary(gui=gui, 
                          obs=DEFAULT_OBS, 
                          act=DEFAULT_ACT)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    print('[INFO] Creating PPO model')
    model = PPO('MlpPolicy',
                train_env,
                verbose=1,
                n_steps=128)

    total_timesteps = int(1e5)  # 100K steps
    
    print(f'[INFO] Starting training for {total_timesteps} timesteps')
    print(f'[INFO] Models will be saved to: {filename}/')
    
    model.learn(total_timesteps=total_timesteps,
                log_interval=100)
    print('[INFO] Training completed successfully')

    #### Save the final model ########################################
    final_model_path = filename+'/final_model.zip'
    model.save(final_model_path)
    print(f'[INFO] Final model saved to: {final_model_path}')
    
    print(filename)
    train_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gate flying training')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder for logs', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
