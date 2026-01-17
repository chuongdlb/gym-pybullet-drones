import numpy as np
import gymnasium as gym
from gym_pybullet_drones.envs.GateAviary import GateAviary

def verify():
    # Create environment
    env = GateAviary(gui=False, record=False)
    
    print("Testing Gate Static Placement...")
    
    episode_positions = []
    
    num_episodes = 3
    for i in range(num_episodes):
        print(f"\nEpisode {i+1}:")
        env.reset(seed=i)
        
        gate_positions = np.array(env.gate_positions)
        episode_positions.append(gate_positions)
        
        print(f"  Gate 0 pos: {gate_positions[0]}")
        
    print("\nComparing Episodes...")
    # Compare all episodes to the first one
    all_match = True
    base_pos = episode_positions[0]
    for i in range(1, num_episodes):
        diff = np.max(np.abs(episode_positions[i] - base_pos))
        if diff > 1e-6:
            print(f"  [FAIL] Episode {i+1} differs from Episode 1 by {diff}")
            all_match = False
        else:
            print(f"  [PASS] Episode {i+1} matches Episode 1.")
            
    if all_match:
        print("\n[SUCCESS] Track is fully static.")
        
    print("\nChecking Gate Parameters:")
    print(f"  Passing Threshold: {env.GATE_PASSING_THRESHOLD}")
    print(f"  Number of Gates: {env.NUM_GATES}")
    
    env.close()

if __name__ == "__main__":
    verify()
