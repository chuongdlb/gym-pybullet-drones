# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

Poetry-based build system (Python 3.10+):
```bash
pip install -e .   # editable install
```

On Ubuntu, may need `sudo apt install build-essential` for pybullet compilation.

## Testing

```bash
pytest tests/                    # run all tests
pytest tests/test_build.py       # import-only tests
pytest tests/test_examples.py    # runs example scripts headless
pytest tests/test_examples.py::test_pid  # single test
```

Tests run in CI on Ubuntu/Python 3.10 via `.github/workflows/test.yml`. Test examples use `gui=False, plot=False, output_folder='tmp'`.

## Architecture

PyBullet-based Gymnasium environments for quadcopter RL. Core inheritance chain:

```
gym.Env
└── BaseAviary          # Physics engine: PyBullet sim, drone URDF loading, state vectors,
    │                   # vision capture, multiple physics modes (PYB, DYN, ground/drag/downwash)
    ├── BaseRLAviary    # RL layer: action/obs space dispatch, action buffer, PID controllers
    │   ├── HoverAviary / MultiHoverAviary   # hover tasks
    │   ├── GateAviary                       # fly through randomized gates
    │   ├── DenseForestAviary                # multi-agent forest navigation (MAVSAC)
    │   ├── CtrlAviary / VelocityAviary      # control/planning tasks
    └── BetaAviary / CFAviary                # firmware-in-the-loop (Betaflight/Crazyflie)
```

**Key enums** (`utils/enums.py`): `DroneModel` (CF2X/CF2P/RACE), `Physics` (6 modes), `ActionType` (RPM/PID/VEL/ONE_D_RPM/ONE_D_PID), `ObservationType` (KIN/RGB).

### Creating a new RL environment

Subclass `BaseRLAviary` and implement 4 abstract methods:
- `_computeReward()` → float
- `_computeTerminated()` → bool
- `_computeTruncated()` → bool
- `_computeInfo()` → dict

Override `_addObstacles()` for custom obstacles, `_observationSpace()`/`_computeObs()` for custom observations. Register in `__init__.py`.

### State vector format (20D per drone)

`_getDroneStateVector(i)` returns: `[pos(3), quat(4), rpy(3), vel(3), ang_vel(3), last_rpm(4)]`

KIN observations extract 12D: `[pos(3), rpy(3), vel(3), ang_vel(3)]` (quaternion skipped).

### Frequency model

`PYB_FREQ` (default 240Hz) controls physics. `CTRL_FREQ` controls policy decisions. `PYB_STEPS_PER_CTRL = PYB_FREQ / CTRL_FREQ`. For RGB observation, `CTRL_FREQ` must satisfy: `(PYB_FREQ / 24) % (PYB_FREQ / CTRL_FREQ) == 0` — use 48 or 24, not 30.

### Training pattern (SB3 PPO)

```python
train_env = make_vec_env(HoverAviary, env_kwargs=dict(obs=..., act=...), n_envs=1)
model = PPO('MlpPolicy', train_env)
model.learn(total_timesteps=1e7, callback=EvalCallback(...))
```

### MAVSAC (Multi-Agent Vision-Sharing Actor-Critic)

Custom algorithm in `algorithms/mavsac.py`. Uses `CrossAttentionVisionExtractor` (SB3 `BaseFeaturesExtractor` subclass) with shared CNN for RGB images + cross-attention between agents + gated fusion. Used with `DenseForestAviary` (Dict observation: vision + state).
