"""GPU-accelerated training with CUDA support.

Uses GPU for neural network training while CPU handles PyBullet physics.
This hybrid approach maximizes utilization of both GPU and CPU.
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_TIMESTEPS = int(1e6)  # 1M steps (can train longer with GPU)

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

def run(output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, 
        timesteps=DEFAULT_TIMESTEPS,
        device='auto'):
    """Train a drone to fly through 6 gates using GPU acceleration."""
    
    # Check CUDA availability
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'[INFO] Using device: {device.upper()}')
    if device == 'cuda':
        print(f'[INFO] GPU: {torch.cuda.get_device_name(0)}')
        print(f'[INFO] CUDA Version: {torch.version.cuda}')
        print(f'[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    filename = os.path.join(output_folder, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    print('[INFO] Creating training environment (CPU-based physics)')
    train_env = GateAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    print(f'[INFO] Creating PPO model with GPU-optimized hyperparameters')
    
    # GPU-optimized PPO hyperparameters
    # With GPU, we can afford:
    # - Larger batch sizes (faster gradient computation on GPU)
    # - Bigger networks (more parameters, better learning)
    # - More training epochs per update
    model = PPO(
        'MlpPolicy',
        train_env,
        device=device,  # Use GPU for neural network
        verbose=1,
        n_steps=512,              # Larger rollout buffer (GPU can handle it)
        batch_size=128,           # Larger minibatch for GPU
        n_epochs=20,              # More gradient updates (GPU is fast)
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.001,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),  # Larger asymmetric network
            ortho_init=True,
            activation_fn=torch.nn.ReLU
        )
    )

    total_timesteps = int(timesteps)
    
    print(f'[INFO] Starting training for {total_timesteps:,} timesteps')
    print(f'[INFO] Models will be saved to: {filename}/')
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=10  # More frequent logging with GPU
        )
        
        elapsed = time.time() - start_time
        print(f'\n[INFO] Training completed successfully in {elapsed/60:.1f} minutes')
        print(f'[INFO] Training speed: {total_timesteps/elapsed:.0f} steps/sec')
        
        if device == 'cuda':
            print(f'[INFO] Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f'\n[INFO] Training interrupted after {elapsed/60:.1f} minutes')
        print(f'[INFO] Completed {model.num_timesteps:,} / {total_timesteps:,} timesteps')

    #### Save the final model ########################################
    final_model_path = filename+'/final_model.zip'
    model.save(final_model_path)
    print(f'[INFO] Final model saved to: {final_model_path}')
    
    # Clear GPU memory
    if device == 'cuda':
        del model
        torch.cuda.empty_cache()
    
    print(filename)
    train_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU-accelerated gate flying training')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder for logs', metavar='')
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int,           help='Total training timesteps', metavar='')
    parser.add_argument('--device',             default='auto',                type=str,           help='Device: auto, cuda, or cpu', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
