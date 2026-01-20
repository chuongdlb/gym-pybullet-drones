"""Script demonstrating the use of `gym_pybullet_drones` for gate flying.

The GateAviary class is used as a learning environment for the PPO algorithm to train
a single drone to fly through 3 randomly positioned gates.

Example
-------
In a terminal, run as:

    $ python learn_gates.py --gui false
    $ python learn_gates.py --gui true

Notes
-----
This script trains a drone to navigate through 3 randomly placed gates using
reinforcement learning with the PPO algorithm from stable-baselines3.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_RESUME_MODEL = None

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, resume_model_path=DEFAULT_RESUME_MODEL):

    filename = os.path.join(output_folder, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # Use limited parallel environments for better performance with PyBullet
    # PyBullet physics simulation doesn't parallelize well on Windows
    num_envs = min(2, multiprocessing.cpu_count())  # Reduced to 2 for Windows stability
    print(f'[INFO] Using {num_envs} parallel environments')

    # Create training environment with DummyVecEnv (better compatibility with Windows)
    # DummyVecEnv runs environments in same process but with vectorization
    train_env = make_vec_env(GateAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                             n_envs=num_envs,
                             seed=0,
                             vec_env_cls=DummyVecEnv
                             )
    # Create eval environment (single env)
    eval_env = make_vec_env(GateAviary,
                           env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                           n_envs=1,
                           seed=0,
                           vec_env_cls=DummyVecEnv
                           )

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Check for existing model to resume training ###########
    existing_model = None
    
    # First, check if user specified a model path
    if resume_model_path:
        if os.path.isfile(resume_model_path):
            existing_model = resume_model_path
            print(f'[INFO] User-specified model found: {existing_model}')
        else:
            print(f'[WARNING] Specified model not found: {resume_model_path}')
            print('[INFO] Will search for most recent model instead...')
    
    # If no user-specified model or it wasn't found, auto-detect latest
    if existing_model is None and os.path.exists(output_folder):
        # Find all gate-* directories
        gate_dirs = glob.glob(os.path.join(output_folder, 'gate-*'))
        if gate_dirs:
            # Sort by modification time to get the most recent
            gate_dirs.sort(key=os.path.getmtime, reverse=True)
            for gate_dir in gate_dirs:
                best_model_path = os.path.join(gate_dir, 'best_model.zip')
                if os.path.isfile(best_model_path):
                    existing_model = best_model_path
                    print(f'[INFO] Auto-detected most recent model: {existing_model}')
                    break

    #### Train the model #######################################
    if existing_model:
        print(f'[INFO] Resuming training from: {existing_model}')
        model = PPO.load(existing_model, env=train_env)
        print(f'[INFO] Model loaded successfully, continuing training...')
    else:
        print('[INFO] No existing model found, starting training from scratch')
        model = PPO('MlpPolicy',
                    train_env,
                    # tensorboard_log=filename+'/tb/',
                    verbose=1)

    #### Target cumulative rewards (problem-dependent) ##########
    # Reward structure for 6 gates with racing focus: 
    # - Pass each gate: +150
    # - Pass all 6 gates: +200 bonus
    # - Speed and progress rewards
    # Target: Successfully pass all 6 gates with racing optimization
    target_reward = 1100.0  # (6 gates * 150) + 200 bonus + speed/progress rewards
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),  # Evaluate more frequently
                                #  n_eval_episodes=5,  # Number of episodes for evaluation
                                 deterministic=True,
                                 render=False)
    
    print(f'[INFO] Starting training...')
    print(f'[INFO] Models will be saved to: {filename}/')
    print(f'[INFO] Evaluation every {eval_callback.eval_freq} steps ({eval_callback.eval_freq * num_envs} timesteps)')
    print(f'[INFO] Best model will be saved when performance improves')
    print(f'[INFO] Target reward threshold: {target_reward}')
    
    model.learn(total_timesteps=int(2e6) if local else int(1e2), # 2M steps for first iteration with racing
                callback=eval_callback,
                log_interval=100)

    #### Save the final model ########################################
    final_model_path = filename+'/final_model.zip'
    model.save(final_model_path)
    print(f'[INFO] Final model saved to: {final_model_path}')
    
    # Also save current model as best if no best_model exists yet
    best_model_path = filename+'/best_model.zip'
    if not os.path.isfile(best_model_path):
        model.save(best_model_path)
        print(f'[INFO] Saved current model as best_model: {best_model_path}')
    print(filename)

    #### Print training progression ############################
    eval_file = filename+'/evaluations.npz'
    if os.path.isfile(eval_file):
        with np.load(eval_file) as data:
            timesteps = data['timesteps']
            results = data['results'][:, 0] 
            print("\nData from evaluations.npz")
            for j in range(timesteps.shape[0]):
                print(f"{timesteps[j]},{results[j]}")
            if local:
                plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
                plt.xlabel('Training Steps')
                plt.ylabel('Episode Reward')
                plt.title('Gate Flying Training Progress')
                plt.grid(True, alpha=0.6)
                plt.show()
    else:
        print(f'[WARNING] Evaluation file not found: {eval_file}')
        print('[INFO] Training may have been too short for evaluations to run')

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    # Load best model for testing
    print("\n[INFO] Loading model for visualization...")
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
        print(f"[INFO] Loading best model: {path}")
    elif os.path.isfile(filename+'/final_model.zip'):
        path = filename+'/final_model.zip'
        print(f"[INFO] Best model not found, loading final model: {path}")
    else:
        print(f"[ERROR]: No model found in {filename}")
        print("[INFO] Skipping visualization")
        return
    
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    test_env = GateAviary(gui=gui,
                         obs=DEFAULT_OBS,
                         act=DEFAULT_ACT,
                         record=record_video)
    test_env_nogui = GateAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, 
              "\tTerminated:", terminated, "\tTruncated:", truncated, 
              "\tInfo:", info)
        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                timestamp=i/test_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3],
                                    np.zeros(4),
                                    obs2[3:15],
                                    act2
                                    ]),
                control=np.zeros(12)
                )
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Gate flying reinforcement learning script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--resume_model_path',  default=DEFAULT_RESUME_MODEL,  type=str,           help='Path to model to resume training from (default: auto-detect latest)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
