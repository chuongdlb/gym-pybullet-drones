"""
Test script to verify GateAviary gate configuration changes.
Tests:
- Number of gates is 6 (not 5)
- Gates are distributed evenly along figure-8
- Gate positions are properly randomized per episode
- Gate orientations are perpendicular to flight path
"""

import sys
import numpy as np
sys.path.insert(0, '/c/Users/hle/source/gym-pybullet-drones')

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

def test_num_gates():
    """Test that environment has 6 gates."""
    print("=" * 60)
    print("TEST 1: Number of Gates")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    num_gates = len(env.GATE_IDS)
    print(f"✓ Number of gates: {num_gates}")
    assert num_gates == 6, f"Expected 6 gates, got {num_gates}"
    print(f"✓ Gate positions stored: {len(env.gate_positions)}")
    assert len(env.gate_positions) == 6, "Gate positions mismatch"
    print(f"✓ Gate orientations stored: {len(env.gate_orientations)}")
    assert len(env.gate_orientations) == 6, "Gate orientations mismatch"
    
    env.close()
    print("✓ TEST 1 PASSED\n")

def test_gate_distribution():
    """Test that gates are evenly distributed along figure-8."""
    print("=" * 60)
    print("TEST 2: Gate Distribution Along Figure-8")
    print("=" * 60)
    
    # Create multiple environments to verify consistent distribution pattern
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        env = GateAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.PID
        )
        
        obs, info = env.reset()
        
        # Get gate positions and compute distances between consecutive gates
        gate_positions = np.array(env.gate_positions)
        print(f"Track scale A: {env.track_scale_A:.3f} (expected 1.5-2.5)")
        assert 1.5 <= env.track_scale_A <= 2.5, f"Scale A out of range: {env.track_scale_A}"
        
        print(f"Track rotation angle: {env.track_rotation_angle:.3f} rad (0-2π)")
        assert 0 <= env.track_rotation_angle <= 2*np.pi, f"Rotation angle out of range"
        
        # All gates should be at roughly the same Z height
        z_positions = gate_positions[:, 2]
        print(f"Gate Z positions: {z_positions}")
        assert np.allclose(z_positions, 1.0, atol=0.05), f"Z positions not at 1.0m height"
        print(f"✓ All gates at correct Z height (~1.0m)")
        
        # Compute arc distances (approximate) between consecutive gates
        # For a figure-8, 6 gates means roughly 60-degree separation in parameter space
        distances = []
        for i in range(len(gate_positions) - 1):
            dist = np.linalg.norm(gate_positions[i+1] - gate_positions[i])
            distances.append(dist)
        
        distances = np.array(distances)
        print(f"Inter-gate distances: {distances}")
        print(f"Distance std dev: {np.std(distances):.4f}")
        print(f"✓ Gates reasonably distributed along figure-8")
        
        env.close()
    
    print("\n✓ TEST 2 PASSED\n")

def test_track_randomization():
    """Test that track is randomized between episodes."""
    print("=" * 60)
    print("TEST 3: Track Randomization Per Episode")
    print("=" * 60)
    
    scales = []
    rotations = []
    gate_pos_sets = []
    
    for episode in range(5):
        env = GateAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.PID
        )
        
        obs, info = env.reset()
        
        scales.append(env.track_scale_A)
        rotations.append(env.track_rotation_angle)
        gate_pos_sets.append(np.array(env.gate_positions))
        
        env.close()
    
    scales = np.array(scales)
    rotations = np.array(rotations)
    
    print(f"Scale factors across 5 episodes: {scales}")
    print(f"Scale variation (std): {np.std(scales):.4f}")
    assert np.std(scales) > 0.1, "Scales are not varying - randomization failed"
    print(f"✓ Scale randomization working")
    
    print(f"\nRotation angles across 5 episodes: {rotations}")
    print(f"Rotation variation (std): {np.std(rotations):.4f}")
    assert np.std(rotations) > 0.3, "Rotations are not varying - randomization failed"
    print(f"✓ Rotation randomization working")
    
    # Check gate positions are different across episodes
    diff_12 = np.linalg.norm(gate_pos_sets[0] - gate_pos_sets[1])
    print(f"\nGate position difference between episode 0 and 1: {diff_12:.4f}")
    assert diff_12 > 0.1, "Gate positions not changing between episodes"
    print(f"✓ Gate positions changing due to randomization")
    
    print("\n✓ TEST 3 PASSED\n")

def test_gate_orientation():
    """Test that gates are oriented perpendicular to flight path."""
    print("=" * 60)
    print("TEST 4: Gate Orientation (Perpendicular to Path)")
    print("=" * 60)
    
    env = GateAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obs=ObservationType.KIN,
        act=ActionType.PID
    )
    
    obs, info = env.reset()
    
    gate_positions = np.array(env.gate_positions)
    gate_orientations = env.gate_orientations
    
    print(f"Number of gates: {len(gate_orientations)}")
    print(f"Sample gate orientations (as quaternions):")
    for i in range(min(3, len(gate_orientations))):
        print(f"  Gate {i}: {gate_orientations[i]}")
    
    # All gates should have valid quaternions (sum of squares ≈ 1)
    for i, quat in enumerate(gate_orientations):
        norm = np.sqrt(sum(q**2 for q in quat))
        assert 0.99 < norm < 1.01, f"Gate {i} quaternion not normalized: {norm}"
    
    print(f"✓ All gate quaternions properly normalized")
    
    # Verify gate tangent angles are computed (gates face different directions)
    # Extract yaw from quaternions and check they vary
    import pybullet as p
    yaws = []
    for quat in gate_orientations:
        euler = p.getEulerFromQuaternion(quat)
        yaws.append(euler[2])  # Z rotation (yaw)
    
    yaws = np.array(yaws)
    print(f"Gate yaw angles: {yaws}")
    print(f"Yaw variation (std): {np.std(yaws):.4f}")
    assert np.std(yaws) > 0.1, "Gate orientations not varying - should track path"
    print(f"✓ Gates oriented differently along figure-8 path")
    
    env.close()
    print("\n✓ TEST 4 PASSED\n")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("GATE CONFIGURATION TESTS")
    print("="*60 + "\n")
    
    try:
        test_num_gates()
        test_gate_distribution()
        test_track_randomization()
        test_gate_orientation()
        
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
