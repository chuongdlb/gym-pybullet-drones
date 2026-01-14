# Gate Flying Training Environment

This document describes the new gate flying training environment created for gym-pybullet-drones.

## Files Created

### 1. GateAviary.py
**Location:** `gym_pybullet_drones/envs/GateAviary.py`

A new RL environment that inherits from `BaseRLAviary` designed for training a single drone to fly through randomly positioned gates.

**Key Features:**
- **3 Random Gates**: Each episode spawns 3 gates at random positions and orientations
- **Gate Positions**: Gates spawn within bounds (x: -2 to 2, y: -2 to 2, z: 0.7 to 1.5)
- **Minimum Spacing**: Gates are at least 1.5m apart from each other and 1m from the drone start position
- **Gate Detection**: Uses position and orientation to detect when drone passes through a gate
- **Episode Length**: 20 seconds

**Reward Structure:**
- `+0.1`: Staying alive per step
- `+0.5 * (3.0 - distance)`: Moving closer to next gate
- `+100`: Successfully passing through a gate
- `+200`: Bonus for passing all 3 gates
- `-1.0`: Penalty for excessive tilting (>0.5 rad)
- `-2.0`: Penalty for going out of bounds

**Termination:**
- **Success**: All 3 gates passed
- **Truncation**: Out of bounds, excessive tilt (>0.8 rad), or timeout

### 2. learn_gates.py
**Location:** `gym_pybullet_drones/examples/learn_gates.py`

Training script based on `learn.py` specifically configured for the gate flying task.

**Configuration:**
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Observation**: Kinematic (KIN)
- **Action**: RPM control
- **Training Steps**: 10M timesteps
- **Target Reward**: 350 (achievable when passing all gates)
- **Eval Frequency**: Every 2000 steps

## Environment Registration

The environment is registered in `gym_pybullet_drones/__init__.py` as:
```python
register(
    id='gate-aviary-v0',
    entry_point='gym_pybullet_drones.envs:GateAviary',
)
```

## Usage

### Training
```bash
cd gym_pybullet_drones/examples/
python learn_gates.py --gui false  # Train without visualization
python learn_gates.py --gui true   # Train with visualization (slower)
```

### Arguments
- `--gui`: Whether to use PyBullet GUI (default: True)
- `--record_video`: Whether to record video (default: False)
- `--output_folder`: Folder for logs (default: "results")
- `--colab`: Whether running in notebook (default: False)

### Testing Trained Model
After training, the script automatically:
1. Evaluates the best model over 10 episodes
2. Visualizes one episode with the GUI
3. Logs flight data
4. Plots training progress

## Observation Space

The observation space is kinematic (size 12 + action buffer):
- Position (x, y, z)
- Orientation (roll, pitch, yaw)
- Linear velocity (vx, vy, vz)
- Angular velocity (wx, wy, wz)
- Previous actions (buffered)

## Action Space

RPM control: 4 values in range [-1, 1] representing normalized RPM commands for each motor.

## Expected Training Time

Training to convergence typically requires:
- **CPU**: 2-4 hours for reasonable performance
- **GPU**: Not required (PPO with MlpPolicy)
- **Episodes**: Several thousand episodes to learn gate navigation

## Tips for Better Performance

1. **Longer Training**: Increase timesteps beyond 10M if not converging
2. **Action Type**: Try `ActionType.VEL` or `ActionType.PID` for smoother control
3. **Gate Spacing**: Adjust `min_gate_distance` in GateAviary for easier/harder tasks
4. **Reward Tuning**: Modify reward values to encourage desired behaviors
5. **Observation**: Consider adding gate positions to observation space for faster learning

## Troubleshooting

**Issue**: Drone crashes immediately
- **Solution**: Reduce initial exploration, adjust reward penalties

**Issue**: Drone doesn't approach gates
- **Solution**: Increase reward for moving toward gates, reduce alive bonus

**Issue**: Training is slow
- **Solution**: Use `--gui false`, increase `n_envs` in make_vec_env()

**Issue**: Gate collisions not detected
- **Solution**: The gates use collision detection via PyBullet, ensure gate.urdf is properly loaded
