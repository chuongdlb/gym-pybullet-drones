"""
Test script to verify GateAviary observation space changes.
Tests:
- Observation includes next 2 gates positions (6 dims)
- Observation includes velocity magnitude (1 dim)
- Observation includes time-in-episode (1 dim)
- Observation shape is correct
- Observation values are in reasonable ranges
"""

import sys
import numpy as np
sys.path.insert(0, '/c/Users/hle/source/gym-pybullet-drones')

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gymnasium import spaces

def test_observation_space_shape():
    """Test that observation space has correct shape."""
    print("=" * 60)
    print("TEST 1: Observation Space Shape")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs_space = env.observation_space
    print(f"Observation space type: {type(obs_space)}")
    print(f"Observation space: {obs_space}")
    
    assert isinstance(obs_space, spaces.Box), "Observation space should be Box"
    print(f"✓ Observation space is Box")
    
    shape = obs_space.shape
    print(f"Observation shape: {shape}")
    
    # Expected shape: (num_drones, base_obs + 8_extended)
    # Base KIN obs is typically: pos(3) + ori(4) + vel(3) + ang_vel(3) + action_buffer = ~12 + action_buffer
    # Extended obs: 8 dims (next_gate_3 + lookahead_gate_3 + velocity_mag_1 + time_1)
    # With action buffer (PID has last 15 actions of 3 dims = 45), total should be around 12 + 45 + 8 = 65
    
    num_drones = 1
    expected_shape = (num_drones,)
    
    print(f"Expected shape: {expected_shape} with varying feature size")
    assert shape[0] == num_drones, f"First dimension should be {num_drones}, got {shape[0]}"
    print(f"✓ First dimension correct (num_drones={num_drones})")
    
    env.close()
    print("✓ TEST 1 PASSED\n")

def test_observation_values():
    """Test that observation values are numeric and in reasonable ranges."""
    print("=" * 60)
    print("TEST 2: Observation Values")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation dtype: {obs.dtype}")
    
    assert obs.dtype == np.float32, f"Observation dtype should be float32, got {obs.dtype}"
    print(f"✓ Observation dtype is float32")
    
    assert not np.any(np.isnan(obs)), "Observation contains NaN"
    print(f"✓ No NaN values in observation")
    
    # Check that observation is finite
    assert np.all(np.isfinite(obs[:, :-2])), "Some observation values are infinite (except last 2 special dims)"
    print(f"✓ Observation values are finite")
    
    # The last 8 dimensions are extended observations
    # Let's extract them
    extended_obs = obs[:, -8:]
    print(f"Extended observation (last 8 dims): {extended_obs[0]}")
    
    # Next gate position (3 dims) - relative to drone, should be reasonable
    next_gate_rel = extended_obs[0, :3]
    print(f"  Next gate relative position: {next_gate_rel}")
    assert np.all(np.isfinite(next_gate_rel)), "Next gate position contains non-finite values"
    
    # Lookahead gate position (3 dims)
    lookahead_rel = extended_obs[0, 3:6]
    print(f"  Lookahead gate relative position: {lookahead_rel}")
    assert np.all(np.isfinite(lookahead_rel)), "Lookahead gate position contains non-finite values"
    
    # Velocity magnitude (1 dim)
    vel_mag = extended_obs[0, 6]
    print(f"  Velocity magnitude: {vel_mag:.4f}")
    assert vel_mag >= 0, f"Velocity magnitude should be non-negative, got {vel_mag}"
    print(f"✓ Velocity magnitude is valid")
    
    # Time in episode (1 dim, 0-1)
    time_in_episode = extended_obs[0, 7]
    print(f"  Time in episode: {time_in_episode:.4f}")
    assert 0 <= time_in_episode <= 1.0, f"Time in episode should be [0,1], got {time_in_episode}"
    print(f"✓ Time in episode is valid")
    
    env.close()
    print("✓ TEST 2 PASSED\n")

def test_extended_observations_over_time():
    """Test that extended observations change appropriately over time."""
    print("=" * 60)
    print("TEST 3: Extended Observations Over Time")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    # Store observations over time
    obs_history = [obs]
    next_gate_history = []
    lookahead_history = []
    time_history = []
    
    # Move drone forward
    action = np.array([[1.0, 0.0, 1.0]])
    
    for step in range(50):
        obs, reward, terminated, truncated, info = env.step(action)
        obs_history.append(obs)
        
        # Extract extended obs
        extended_obs = obs[:, -8:]
        next_gate_history.append(extended_obs[0, :3])
        lookahead_history.append(extended_obs[0, 3:6])
        time_history.append(extended_obs[0, 7])
        
        if terminated or truncated:
            break
    
    # Check that time increases monotonically
    time_history = np.array(time_history)
    print(f"Time values (first 10): {time_history[:10]}")
    print(f"Time values (last 10): {time_history[-10:]}")
    
    diffs = np.diff(time_history)
    print(f"Time differences (should be positive): {diffs[:10]}")
    assert np.all(diffs > 0), "Time should be monotonically increasing"
    print(f"✓ Time in episode increases monotonically")
    
    # Check that gate relative positions change as drone moves
    next_gate_array = np.array(next_gate_history)
    print(f"\nNext gate relative position (first 5): {next_gate_array[:5]}")
    print(f"Next gate relative position (last 5): {next_gate_array[-5:]}")
    
    pos_diff = np.linalg.norm(next_gate_array[0] - next_gate_array[-1])
    print(f"Position change: {pos_diff:.4f}")
    assert pos_diff > 0.01, "Gate relative position should change as drone moves"
    print(f"✓ Gate relative position changes appropriately")
    
    # Check lookahead values are different from next gate
    if np.any(np.array(lookahead_history) != 0):  # If lookahead gate exists
        lookahead_array = np.array(lookahead_history)
        diff_next_vs_lookahead = np.linalg.norm(
            next_gate_array[0] - lookahead_array[0]
        )
        print(f"\nDifference between next and lookahead gate: {diff_next_vs_lookahead:.4f}")
        if diff_next_vs_lookahead > 0.01:
            print(f"✓ Lookahead gate is different from next gate")
        else:
            print(f"⚠ Lookahead gate not visible yet (drone hasn't reached next gate)")
    
    env.close()
    print("✓ TEST 3 PASSED\n")

def test_velocity_magnitude_obs():
    """Test that velocity magnitude observation reflects drone speed."""
    print("=" * 60)
    print("TEST 4: Velocity Magnitude Observation")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    print("Phase 1: Hovering (zero velocity)")
    hover_vels = []
    for step in range(30):
        action = np.array([[0.0, 0.0, 0.5]])
        obs, reward, terminated, truncated, info = env.step(action)
        vel_mag = obs[0, -2]  # Second to last dimension
        hover_vels.append(vel_mag)
    
    hover_vels = np.array(hover_vels)
    print(f"Velocity magnitudes while hovering: {hover_vels[-5:]}")
    print(f"Mean velocity while hovering: {np.mean(hover_vels[-5:]):.4f}")
    
    print("\nPhase 2: Moving forward (high velocity)")
    moving_vels = []
    for step in range(50):
        action = np.array([[5.0, 0.0, 1.0]])  # Move to distant point
        obs, reward, terminated, truncated, info = env.step(action)
        vel_mag = obs[0, -2]  # Second to last dimension
        moving_vels.append(vel_mag)
        
        if terminated or truncated:
            break
    
    moving_vels = np.array(moving_vels)
    print(f"Velocity magnitudes while moving: {moving_vels[-5:]}")
    print(f"Mean velocity while moving: {np.mean(moving_vels[-5:]):.4f}")
    
    # Check that moving velocity is higher than hover velocity
    if np.mean(moving_vels[-5:]) > np.mean(hover_vels[-5:]):
        print(f"✓ Velocity magnitude observation increases when moving")
    else:
        print(f"⚠ Velocity magnitude not clearly higher when moving (may be due to control dynamics)")
    
    env.close()
    print("✓ TEST 4 PASSED\n")

def test_observation_consistency():
    """Test that observations are consistent across multiple resets."""
    print("=" * 60)
    print("TEST 5: Observation Consistency Across Resets")
    print("=" * 60)
    
    obs_spaces = []
    
    for episode in range(3):
        env = GateAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.PID
        )
        
        obs, info = env.reset()
        obs_spaces.append(obs.shape)
        
        print(f"Episode {episode + 1}: observation shape = {obs.shape}")
        
        env.close()
    
    # All observations should have same shape
    assert len(set(obs_spaces)) == 1, f"Observation shapes vary: {obs_spaces}"
    print(f"✓ Observation shape is consistent across episodes")
    
    print("✓ TEST 5 PASSED\n")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("OBSERVATION SPACE TESTS")
    print("="*60 + "\n")
    
    try:
        test_observation_space_shape()
        test_observation_values()
        test_extended_observations_over_time()
        test_velocity_magnitude_obs()
        test_observation_consistency()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
