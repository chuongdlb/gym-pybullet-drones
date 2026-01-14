import gymnasium as gym
import numpy as np
from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.enums import ObservationType

def test_gate_aviary():
    print("Initializing GateAviary...")
    env = GateAviary(obs=ObservationType.KIN, gui=False, record=False)
    
    # Check observation space dimensions
    # Base observation is 12 + (ACTION_BUFFER_SIZE * 4)
    # We added 3
    base_size = 12 + (env.ACTION_BUFFER_SIZE * 4)
    total_size = base_size + 3
    expected_shape = (1, total_size)
    
    print(f"Action Buffer Size: {env.ACTION_BUFFER_SIZE}")
    print(f"Base Size: {base_size}")
    print(f"Expected Total Size: {total_size}")
    
    if env.observation_space.shape == expected_shape:
        print(f"[PASS] Observation space shape matches expected: {expected_shape}")
    else:
        print(f"[FAIL] Observation space shape mismatch. Expected {expected_shape}, got {env.observation_space.shape}")
    
    # Reset env
    print("\nResetting environment...")
    obs, info = env.reset(seed=42)
    
    # Check observation shape
    print("Observation shape:", obs.shape)
    if obs.shape == expected_shape:
        print("[PASS] Observation shape matches expected.")
    else:
        print(f"[FAIL] Observation shape mismatch. Got {obs.shape}")
        
    # Check values
    # Get ground truth
    gate_pos = env.gate_positions[0]
    state = env._getDroneStateVector(0)
    drone_pos = state[0:3]
    expected_rel_pos = gate_pos - drone_pos
    
    # Get obs values (last 3 elements)
    obs_rel_pos = obs[0, -3:]
    
    print(f"\nGate Position: {gate_pos}")
    print(f"Drone Position: {drone_pos}")
    print(f"Expected Rel Pos: {expected_rel_pos}")
    print(f"Observed Rel Pos: {obs_rel_pos}")
    
    if np.allclose(expected_rel_pos, obs_rel_pos, atol=1e-5):
        print("[PASS] Relative position in observation is correct!")
    else:
        print("[FAIL] Relative position mismatch!")
        
    env.close()

if __name__ == "__main__":
    test_gate_aviary()
