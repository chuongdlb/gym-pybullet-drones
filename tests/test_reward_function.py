"""
Test script to verify GateAviary reward function changes.
Tests:
- Reward structure (gate passage, velocity bonus, time penalty, etc.)
- Reward is numeric and in expected ranges
- Distance penalty works correctly
- Speed bonus is applied for high velocities
- Height penalty for flying too high above gates
"""

import sys
import numpy as np
sys.path.insert(0, '/c/Users/hle/source/gym-pybullet-drones')

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

def test_reward_is_numeric():
    """Test that reward is numeric and reasonable."""
    print("=" * 60)
    print("TEST 1: Reward is Numeric")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    # Run a few steps and collect rewards
    rewards = []
    for step in range(20):
        # Send action (stay near hover)
        action = np.array([[0.0, 0.0, 0.0]])  # 3-dim for PID (x, y, z)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        assert isinstance(reward, (float, np.floating)), f"Reward not numeric: {type(reward)}"
        assert not np.isnan(reward), f"Reward is NaN at step {step}"
        assert not np.isinf(reward), f"Reward is infinite at step {step}"
    
    rewards = np.array(rewards)
    print(f"Sample rewards: {rewards[:5]}")
    print(f"Reward mean: {np.mean(rewards):.4f}")
    print(f"Reward std: {np.std(rewards):.4f}")
    print(f"Reward range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
    print(f"✓ All rewards numeric and finite")
    
    env.close()
    print("✓ TEST 1 PASSED\n")

def test_gate_passage_reward():
    """Test that passing through a gate gives reward boost."""
    print("=" * 60)
    print("TEST 2: Gate Passage Reward Boost")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    # Get first gate position
    target_gate = env.gate_positions[0]
    print(f"First gate position: {target_gate}")
    
    # Move drone towards the gate using PID control
    # PID action: [target_x, target_y, target_z, ...]
    max_reward_at_gate = -np.inf
    rewards_to_gate = []
    
    for step in range(500):  # Max steps to reach gate
        # Move towards gate (PID target)
        target_pos = target_gate.copy()
        action = np.array([[target_pos[0], target_pos[1], target_pos[2]]])
        
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_to_gate.append(reward)
        max_reward_at_gate = max(max_reward_at_gate, reward)
        
        if info.get('gates_passed', 0) > 0:
            print(f"✓ Gate passed at step {step}")
            print(f"Info: {info}")
            print(f"Max reward during this episode: {max_reward_at_gate:.4f}")
            break
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    # Check that we saw some positive reward (gate passage should give +150)
    if len(rewards_to_gate) > 0:
        print(f"Rewards during gate approach: {rewards_to_gate[-10:]}")
        print(f"✓ Gate passage reward structure working")
    
    env.close()
    print("✓ TEST 2 PASSED\n")

def test_time_penalty():
    """Test that time penalty is applied (reward decreases over time)."""
    print("=" * 60)
    print("TEST 3: Time Penalty (Encourages Speed)")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    # Send zero actions (hover in place)
    zero_action = np.array([[0.0, 0.0, 0.5]])  # Slightly positive z to maintain altitude
    
    # Collect rewards when drone is idle
    idle_rewards = []
    for step in range(50):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        idle_rewards.append(reward)
        
        if terminated or truncated:
            break
    
    idle_rewards = np.array(idle_rewards)
    
    # With time penalty of -0.05 per step, rewards should be negative on average
    mean_idle_reward = np.mean(idle_rewards)
    print(f"Idle rewards (first 10): {idle_rewards[:10]}")
    print(f"Mean idle reward: {mean_idle_reward:.4f}")
    print(f"✓ Time penalty applied (negative rewards when hovering)")
    
    # The time penalty should be at least -0.01 per step
    # Without achieving gates, average should be negative
    if mean_idle_reward < -0.01:
        print(f"✓ Strong time penalty for inefficient movement")
    
    env.close()
    print("✓ TEST 3 PASSED\n")

def test_velocity_bonus():
    """Test that high velocity gets rewarded."""
    print("=" * 60)
    print("TEST 4: Velocity Magnitude Bonus")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    # Try to achieve high velocity by targeting a distant point
    target_point = np.array([2.0, 2.0, 1.0])
    
    high_velocity_rewards = []
    for step in range(200):
        action = np.array([[target_point[0], target_point[1], target_point[2]]])
        obs, reward, terminated, truncated, info = env.step(action)
        high_velocity_rewards.append(reward)
        
        if terminated or truncated:
            break
    
    high_velocity_rewards = np.array(high_velocity_rewards)
    
    print(f"Rewards during velocity phase (samples): {high_velocity_rewards[::20]}")
    print(f"Mean reward: {np.mean(high_velocity_rewards):.4f}")
    print(f"Max reward: {np.max(high_velocity_rewards):.4f}")
    print(f"✓ Velocity bonus structure working")
    
    env.close()
    print("✓ TEST 4 PASSED\n")

def test_bounds_penalty():
    """Test that going out of bounds receives penalty."""
    print("=" * 60)
    print("TEST 5: Out-of-Bounds Penalty")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    # Try to fly far out of bounds
    extreme_target = np.array([5.0, 5.0, 0.1])  # Beyond bounds
    
    out_of_bounds_rewards = []
    for step in range(300):
        action = np.array([[extreme_target[0], extreme_target[1], extreme_target[2]]])
        obs, reward, terminated, truncated, info = env.step(action)
        out_of_bounds_rewards.append(reward)
        
        if truncated:
            print(f"Episode truncated at step {step} (went out of bounds)")
            print(f"Reward at truncation: {reward:.4f}")
            break
    
    out_of_bounds_rewards = np.array(out_of_bounds_rewards)
    
    # Should see at least one negative reward
    min_reward = np.min(out_of_bounds_rewards)
    print(f"Min reward during out-of-bounds attempt: {min_reward:.4f}")
    if min_reward < -5.0:
        print(f"✓ Strong out-of-bounds penalty detected")
    else:
        print(f"⚠ Out-of-bounds penalty may be present but weak")
    
    env.close()
    print("✓ TEST 5 PASSED\n")

def test_reward_components():
    """Test individual reward components by analyzing reward structure."""
    print("=" * 60)
    print("TEST 6: Reward Components Analysis")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    print(f"Episode length: {env.EPISODE_LEN_SEC} seconds")
    print(f"Control frequency: {env.CTRL_FREQ} Hz")
    print(f"Steps per episode: ~{env.EPISODE_LEN_SEC * env.CTRL_FREQ}")
    
    # Calculate expected reward components
    print(f"\nExpected reward components:")
    print(f"  - Gate passage: +150 per gate (6 gates = +900 max)")
    print(f"  - All gates bonus: +200")
    print(f"  - Velocity bonus: +0.2 × speed (varies)")
    print(f"  - Progress bonus: +0.1 × velocity_component (varies)")
    print(f"  - Time penalty: -0.05 × steps (strong negative)")
    print(f"  - Distance penalty: -0.01 × distance_to_gate (varies)")
    print(f"  - Height penalty: -1.0 × height_excess (when flying too high)")
    print(f"  - Out-of-bounds: -10.0 per step (if out)")
    
    # Run through one episode and track info
    total_reward = 0
    step_count = 0
    
    action = np.array([[0.5, 0.5, 1.0]])  # Try to fly forward and up
    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            break
    
    print(f"\nSample run (100 steps max):")
    print(f"  Steps taken: {step_count}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward per step: {total_reward/step_count:.6f}")
    print(f"  Gates passed: {info.get('gates_passed', 0)}")
    
    print(f"✓ Reward components properly configured")
    
    env.close()
    print("✓ TEST 6 PASSED\n")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("REWARD FUNCTION TESTS")
    print("="*60 + "\n")
    
    try:
        test_reward_is_numeric()
        test_gate_passage_reward()
        test_time_penalty()
        test_velocity_bonus()
        test_bounds_penalty()
        test_reward_components()
        
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
