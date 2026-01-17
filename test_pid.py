import gymnasium as gym
import numpy as np
from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

try:
    print("Testing PID ActionType...")
    # Initialize with PID action type
    env = GateAviary(gui=False, record=False, act=ActionType.PID)
    
    print("Environment initialized.")
    print(f"Action Space: {env.action_space}")
    
    obs, info = env.reset()
    print("Environment reset.")
    
    # PID action is likely 4D (Target Roll, Pitch, Yaw, Thrust) or similar
    # Check action space shape
    action = env.action_space.sample()
    print(f"Sample Action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print("Step successful.")
    print(f"Reward: {reward}")
    
    env.close()
    print("[SUCCESS] PID ActionType works.")
    
except Exception as e:
    print(f"[ERROR] Failed with PID: {e}")
    import traceback
    traceback.print_exc()
