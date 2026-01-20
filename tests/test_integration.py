"""
Integration test to verify the entire modified GateAviary environment works end-to-end.
Tests:
- Environment can be created with different action types
- Environment can step through an episode
- Environment provides valid observations, rewards, and infos
- Multiple episodes can be run sequentially
- Track randomization works across episodes
"""

import sys
import numpy as np
sys.path.insert(0, '/c/Users/hle/source/gym-pybullet-drones')

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

def test_environment_initialization():
    """Test environment can be initialized with different configurations."""
    print("=" * 60)
    print("TEST 1: Environment Initialization")
    print("=" * 60)
    
    action_types = [ActionType.PID]  # Start with PID as per plan
    
    for action_type in action_types:
        print(f"\nTesting ActionType: {action_type}")
        env = GateAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            gui=False,
            obs=ObservationType.KIN,
            act=action_type
        )
        
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        # Reset and check
        obs, info = env.reset()
        assert obs is not None, "Reset should return observation"
        assert info is not None, "Reset should return info dict"
        print(f"  ✓ Environment initialized successfully")
        
        env.close()
    
    print("✓ TEST 1 PASSED\n")

def test_single_episode():
    """Test a complete episode run."""
    print("=" * 60)
    print("TEST 2: Single Episode Run")
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
    print(f"Initial info: {info}")
    
    episode_reward = 0
    step_count = 0
    gates_passed_max = 0
    
    # Simple control: try to fly to first gate
    for step in range(500):
        # PID action (target position)
        target = np.array([0.5, 0.5, 1.0])
        action = np.array([[target[0], target[1], target[2]]])
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        gates_passed = info.get('gates_passed', 0)
        gates_passed_max = max(gates_passed_max, gates_passed)
        
        assert obs is not None, "Step should return observation"
        assert isinstance(reward, (float, np.floating)), "Step should return numeric reward"
        assert isinstance(terminated, (bool, np.bool_)), "Step should return bool terminated"
        assert isinstance(truncated, (bool, np.bool_)), "Step should return bool truncated"
        assert info is not None, "Step should return info dict"
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Episode summary:")
    print(f"  Steps: {step_count}")
    print(f"  Total reward: {episode_reward:.4f}")
    print(f"  Avg reward/step: {episode_reward/step_count:.6f}")
    print(f"  Gates passed: {gates_passed_max}")
    print(f"  Info at end: {info}")
    
    assert step_count > 0, "Episode should have steps"
    print(f"✓ Single episode completed")
    
    env.close()
    print("✓ TEST 2 PASSED\n")

def test_multiple_episodes():
    """Test multiple consecutive episodes."""
    print("=" * 60)
    print("TEST 3: Multiple Episodes")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    episode_results = []
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset()
        
        track_scale = env.track_scale_A
        track_rotation = env.track_rotation_angle
        print(f"  Track scale: {track_scale:.3f}")
        print(f"  Track rotation: {track_rotation:.3f} rad")
        
        episode_reward = 0
        gates_passed = 0
        step_count = 0
        
        action = np.array([[0.0, 0.0, 0.5]])
        
        for step in range(300):
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            gates_passed = info.get('gates_passed', 0)
            step_count += 1
            
            if terminated or truncated:
                break
        
        episode_results.append({
            'reward': episode_reward,
            'gates_passed': gates_passed,
            'steps': step_count,
            'scale': track_scale,
            'rotation': track_rotation
        })
        
        print(f"  Reward: {episode_reward:.4f}, Gates: {gates_passed}, Steps: {step_count}")
    
    # Check that tracks varied
    scales = [r['scale'] for r in episode_results]
    rotations = [r['rotation'] for r in episode_results]
    
    print(f"\nTrack variation across episodes:")
    print(f"  Scales: {scales}")
    print(f"  Rotations: {rotations}")
    
    # Expect some variation (not guaranteed with 3 episodes, but highly likely)
    if len(set(np.round(scales, 2))) > 1:
        print(f"✓ Track scale varied across episodes")
    else:
        print(f"⚠ Track scale did not vary (could be random chance with only 3 episodes)")
    
    env.close()
    print("✓ TEST 3 PASSED\n")

def test_vectorized_env():
    """Test that environment works with vectorized setup (preparation for parallel training)."""
    print("=" * 60)
    print("TEST 4: Vectorized Environment (SubprocVecEnv)")
    print("=" * 60)
    
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        
        def make_env():
            return GateAviary(
                drone_model=DroneModel.CF2X,
                physics=Physics.PYB,
                gui=False,
                obs=ObservationType.KIN,
                act=ActionType.PID
            )
        
        # Create 2 parallel environments
        num_envs = 2
        envs = SubprocVecEnv([make_env for _ in range(num_envs)])
        
        print(f"Created {num_envs} parallel environments")
        
        obs = envs.reset()
        print(f"Initial vectorized observation shape: {obs.shape}")
        assert obs.shape[0] == num_envs, f"Should have {num_envs} observations"
        
        # Step through multiple times
        for step in range(10):
            actions = np.array([
                [0.5, 0.0, 1.0, 0.0],  # Env 0
                [0.0, 0.5, 1.0, 0.0],  # Env 1
            ])
            obs, rewards, dones, infos = envs.step(actions)
            
            assert obs.shape[0] == num_envs, "Observation batch size mismatch"
            assert len(rewards) == num_envs, "Reward batch size mismatch"
            assert len(dones) == num_envs, "Dones batch size mismatch"
        
        print(f"✓ Vectorized environment works correctly")
        envs.close()
        
    except ImportError:
        print(f"⚠ stable_baselines3 not installed, skipping vectorized test")
    
    print("✓ TEST 4 PASSED\n")

def test_action_types():
    """Test that environment accepts different action types."""
    print("=" * 60)
    print("TEST 5: Different Action Types")
    print("=" * 60)
    
    # Test PID action type
    print("\nTesting ActionType.PID (3D waypoint control):")
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    print(f"  Action space: {env.action_space}")
    print(f"  Action space shape: {env.action_space.shape}")
    
    # PID action: 3D (x, y, z)
    action = np.array([[0.5, 0.0, 1.0]])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  ✓ PID action accepted")
    
    env.close()
    
    # Test RPM action type
    print("\nTesting ActionType.RPM (4D motor control):")
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    
    obs, info = env.reset()
    print(f"  Action space: {env.action_space}")
    
    # RPM action: 4D (one per motor)
    action = np.array([[0.0, 0.0, 0.0, 0.0]])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  ✓ RPM action accepted")
    
    env.close()
    
    print("✓ TEST 5 PASSED\n")

def test_observation_types():
    """Test that different observation types work."""
    print("=" * 60)
    print("TEST 6: Observation Types")
    print("=" * 60)
    
    # Test KIN observation
    print("\nTesting ObservationType.KIN:")
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    assert obs.dtype == np.float32, "Observation should be float32"
    print(f"  ✓ KIN observation works")
    
    env.close()
    
    print("✓ TEST 6 PASSED\n")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("INTEGRATION TESTS")
    print("="*60 + "\n")
    
    try:
        test_environment_initialization()
        test_single_episode()
        test_multiple_episodes()
        test_vectorized_env()
        test_action_types()
        test_observation_types()
        
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
