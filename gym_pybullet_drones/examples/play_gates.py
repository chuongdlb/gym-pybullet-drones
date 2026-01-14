"""Script to visualize and test trained gate flying models.

This script loads the best trained model from learn_gates.py and visualizes
the drone flying through randomly placed gates.

Example
-------
In a terminal, run as:

    $ python play_gates.py
    $ python play_gates.py --model_path results/gate-01.12.2026_10.30.45/best_model.zip

Notes
-----
If no model path is specified, the script will automatically load the most
recent best_model.zip from the results folder.

"""
import os
import time
import argparse
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_MODEL_PATH = None
DEFAULT_NUM_EPISODES = 3

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

def find_latest_model(output_folder='results'):
    """Find the most recent best_model.zip in the results folder.
    
    Parameters
    ----------
    output_folder : str
        The folder containing training results.
        
    Returns
    -------
    str or None
        Path to the most recent best_model.zip, or None if not found.
    """
    if not os.path.exists(output_folder):
        return None
    
    # Find all gate-* directories
    gate_dirs = glob.glob(os.path.join(output_folder, 'gate-*'))
    if not gate_dirs:
        return None
    
    # Sort by modification time to get the most recent
    gate_dirs.sort(key=os.path.getmtime, reverse=True)
    
    for gate_dir in gate_dirs:
        best_model_path = os.path.join(gate_dir, 'best_model.zip')
        if os.path.isfile(best_model_path):
            return best_model_path
    
    return None

def run(model_path=DEFAULT_MODEL_PATH, 
        gui=DEFAULT_GUI, 
        plot=DEFAULT_PLOT, 
        record_video=DEFAULT_RECORD_VIDEO,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        num_episodes=DEFAULT_NUM_EPISODES):
    """Run visualization of trained model.
    
    Parameters
    ----------
    model_path : str, optional
        Path to the trained model. If None, finds the most recent model.
    gui : bool, optional
        Whether to display PyBullet GUI.
    plot : bool, optional
        Whether to plot flight data after completion.
    record_video : bool, optional
        Whether to record video of the flight.
    output_folder : str, optional
        Folder containing training results.
    num_episodes : int, optional
        Number of episodes to run.
    """
    
    #### Find and load the model ################################
    if model_path is None:
        model_path = find_latest_model(output_folder)
        if model_path is None:
            print(f"[ERROR] No trained model found in '{output_folder}/' folder.")
            print("Please train a model first using learn_gates.py or specify a model path.")
            return
        print(f"[INFO] Loading most recent model: {model_path}")
    else:
        if not os.path.isfile(model_path):
            print(f"[ERROR] Model not found at: {model_path}")
            return
        print(f"[INFO] Loading specified model: {model_path}")
    
    try:
        model = PPO.load(model_path)
        print("[INFO] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    #### Create test environments ###############################
    test_env = GateAviary(gui=gui,
                         obs=DEFAULT_OBS,
                         act=DEFAULT_ACT,
                         record=record_video)
    test_env_nogui = GateAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    
    #### Evaluate the model ####################################
    print("\n[INFO] Evaluating model performance over 10 episodes (no GUI)...")
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10,
                                              deterministic=True)
    print(f"[INFO] Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
    
    #### Setup logger ##########################################
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                   num_drones=1,
                   output_folder=output_folder)
    
    #### Run visualization episodes ############################
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Running Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        obs, info = test_env.reset(seed=42+episode, options={})
        start = time.time()
        episode_reward = 0
        gates_passed = 0
        
        for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            episode_reward += reward
            gates_passed = info.get('gates_passed', 0)
            
            obs2 = obs.squeeze()
            act2 = action.squeeze()
            
            # Log state for plotting
            if DEFAULT_OBS == ObservationType.KIN and episode == 0:  # Only log first episode
                logger.log(drone=0,
                          timestamp=i/test_env.CTRL_FREQ,
                          state=np.hstack([obs2[0:3],
                                          np.zeros(4),
                                          obs2[3:15],
                                          act2]),
                          control=np.zeros(12))
            
            test_env.render()
            sync(i, start, test_env.CTRL_TIMESTEP)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} Complete:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Gates Passed: {gates_passed}/{test_env.NUM_GATES}")
        print(f"  Success: {'YES' if gates_passed == test_env.NUM_GATES else 'NO'}")
    
    test_env.close()
    test_env_nogui.close()
    
    #### Plot results ##########################################
    if plot and DEFAULT_OBS == ObservationType.KIN:
        print("\n[INFO] Generating flight plots...")
        logger.plot()

if __name__ == '__main__':
    #### Define and parse arguments ############################
    parser = argparse.ArgumentParser(description='Play trained gate flying model')
    parser.add_argument('--model_path',     default=DEFAULT_MODEL_PATH,    type=str,       help='Path to trained model (default: auto-detect latest)', metavar='')
    parser.add_argument('--gui',            default=DEFAULT_GUI,           type=str2bool,  help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot',           default=DEFAULT_PLOT,          type=str2bool,  help='Whether to plot flight data (default: True)', metavar='')
    parser.add_argument('--record_video',   default=DEFAULT_RECORD_VIDEO,  type=str2bool,  help='Whether to record video (default: False)', metavar='')
    parser.add_argument('--output_folder',  default=DEFAULT_OUTPUT_FOLDER, type=str,       help='Folder with training results (default: "results")', metavar='')
    parser.add_argument('--num_episodes',   default=DEFAULT_NUM_EPISODES,  type=int,       help='Number of episodes to run (default: 3)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
