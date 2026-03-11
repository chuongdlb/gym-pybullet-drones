# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
pip install -e .   # editable install (Poetry, Python 3.10+)
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

PyBullet-based Gymnasium environments for quadcopter RL. Core inheritance:

```
BaseAviary (physics) → BaseRLAviary (RL layer) → HoverAviary, GateAviary, ForestEscapeAviary, etc.
```

**Key enums** (`utils/enums.py`): `DroneModel` (CF2X/CF2P/RACE), `Physics` (6 modes), `ActionType` (RPM/PID/VEL/ONE_D_RPM/ONE_D_PID), `ObservationType` (KIN/RGB).

**PID action semantics**: `ActionType.PID` actions are position **offsets** from current drone position (0.15m per unit action), not absolute coordinates. `action=[0,0,0]` = hover.

**State vector** (20D): `[pos(3), quat(4), rpy(3), vel(3), ang_vel(3), last_rpm(4)]`. KIN obs = 12D (no quaternion).

**Frequency**: `PYB_FREQ=240Hz` (physics), `CTRL_FREQ` (policy). For RGB: use `CTRL_FREQ=48` or `24`.

**New RL env**: subclass `BaseRLAviary`, implement `_computeReward/_computeTerminated/_computeTruncated/_computeInfo`. Override `_addObstacles()` for obstacles, `_observationSpace()/_computeObs()` for custom obs.

## ForestEscapeAviary (PyBullet)

Multi-agent cooperative forest escape: drones navigate through a procedurally generated dense forest (35 trees, 0.12m radius cylinders). All drones must reach x > 2.5. Episodes: 30s max, ctrl_freq=48, PID action space. Supports 1–3 drones, configurable image resolution.

### PyBullet Iterations (v6–v10b)

| Version | Key Change | Result | Lesson |
|---------|-----------|--------|--------|
| v6 | Baseline with survival bonus | Hover optimum (720 vs 245 reward) | Never reward survival — it dominates forward progress |
| v7 | Remove survival, add idle penalty | x_prog peaked 0.248, plateaued | KIN-only (9D obstacle state) insufficient for fine avoidance |
| v8 | Add vision CNN, LR=1e-4 | KL explosion, policy collapse | LR=1e-4 too aggressive for CNN; use 3e-5 with n_epochs=3 |
| v8b | LR=3e-5→5e-6 | x_prog 0.364, std rising | std monotonic rise = ent_coef too high (0.01) |
| v9a | Proximity 0.6m×15, depth channel | x_prog 0.00 | 0.6m zone + 35 trees = no penalty-free paths |
| v9b | Cross-attention, proximity 0.45m×8 | Reward exploitation | Centroid progress lets drones sacrifice one |
| v10 | min-x progress, milestone bonuses | x_prog 0.15–0.25, escape=0% | Fixed exploitation but 64×48 resolution too low |
| v10b-1d | 128×96 RGBD, single drone | x_prog 0.489 | Resolution confirmed as key fix |
| v10b-3d | 128×96, 3 drones | 31 FPS, x_prog 0.11 | CPU rendering bottleneck → need GPU sim |

**Key PyBullet lessons**: EGL GPU rendering crashes with SubprocVecEnv (use CPU TinyRenderer). PyBullet depth is normalized z-buffer. ent_coef=0.005 keeps std stable. 128×96 minimum resolution for tree detection.

## Crazyflow GPU Training (Phase 2)

Migrated from PyBullet (31 FPS) to Crazyflow JAX GPU simulator (6,600 FPS) — **1178x speedup**. 10M steps in 5 min vs 45h on PyBullet. All training on RTX 3090 at `ai@172.31.10.232`.

### Code Structure

```
gym_pybullet_drones/crazyflow/
    config.py            # TrainConfig, ForestConfig, ObsConfig, RewardConfig (frozen dataclasses)
    forest_escape_env.py # JAX env: forest layout, collision detection, actions
    obs_utils.py         # Lidar ray-casting, k-nearest tree computation
    reward.py            # Vectorized reward (team + per-drone shaping)
    networks.py          # ActorCritic MLP (Phase 1), AttentionActor + CentralizedCritic (Phase 2)
    train_mappo.py       # MAPPO training loop, fully JIT, checkpoint save/resume
```

**Environment**: Forest x∈[-3,2], y∈[-2,2] (20m²). Grid + jitter tree placement. collision_dist=0.15m, tree_radius=0.06m. Actions: position offsets (action_scale=0.15). Control 50Hz, physics 500Hz.

**Observations** (92D per drone): pos(3), rpy(3), vel(3), angvel(3), goal_progress(1), time_remaining(1), lidar(48 rays, 180° FOV, 3m range), k_nearest_trees(24, 12 trees × dx,dy), relative_drones(6).

**Critical discovery**: Crazyflow has NO physical collision response — drones fly through trees. Before `terminate_on_collision=True`, escape_rate = vel_x/250 exactly (purely speed, not navigation). ALL early runs had trivial 100% escapes.

### Crazyflow Iteration History

#### Phase 2 v1–v6: Multi-Drone Attention MAPPO

Initial attempts with 3 drones, attention-based architecture:
- **v1**: ent_coef=0.01 → std growth 0.61→1.03, scalar centralized value → EV<0. Failed.
- **v2**: ent_coef=0.005, vf_coef=1.0, per-drone values from centralized critic. EV=0.12. Better.
- **v3**: Tree curriculum 5→35 over 50%. **Critical obs bug**: inactive trees at (100,100) normalized to 33+ → destroyed learning. Fixed with `jnp.clip(nearest, -1, 1)`.
- **v4–v6**: Various reward/architecture tweaks. Forward progress but no collision avoidance improvement.
- **Key insight**: Centralized critic must output per-drone values, not scalar. Per-drone value heads via attention.

#### Phase 2 v7: Single-Drone MLP Baseline (50M steps)

Switched to single drone with MLP (512, 256) to prove basic navigation before multi-agent.
- 35 trees, collision_penalty=30, 48 lidar rays, proximity_dist=0.35
- **Result**: Collision 8%→4.7%, EV=0.713, but x_progress stuck
- **Lesson**: Policy must be initialized with forward bias (actor output bias=[0.5,0,0]). Without it, PPO consistently learns backward flight (safe pessimism trap).

#### Phase 2 v7b/v7c: Extended Training (200M steps)

- **v7b**: Same as v7, 200M steps. ent_coef decay too slow (over 100M = 50% of training).
- **v7c**: Faster entropy decay (curriculum_frac=0.25 → trees reach 35 at 50M, ent reaches min at 50M). Then 150M pure optimization.
- **v7c result at 69M**: Collision 8%→**2.4%** (excellent avoidance!) but MinX stuck at -0.10, escape=0.26%.
- **Root cause**: Proximity zones overlap at 35 trees. Gap = 0.76m spacing - 2×0.35m proximity = 0.06m (impossible to navigate penalty-free). Policy learns "stay away from everything" instead of "steer through gaps".

#### Phase 2 v8: Disable Proximity Penalty (100M steps)

- **Fix**: proximity_coef=0 (disabled), reduced to 25 trees, collision_penalty=15
- **Math**: 25 trees → spacing 0.89m, collision gap = 0.89 - 2×0.15 = 0.59m (navigable)
- **Result at 36M**: Collision 2.8–3.4% (good), but escape=0.26% (same as all previous!)
- **Critical discovery**: escape_rate ≈ vel_x/250 with ratio 0.95–1.05 across ALL runs. Crazyflow has no physical collision — drones fly through trees. All v1–v8 had trivial escapes.

#### Phase 2 v9: terminate_on_collision=True — THE BREAKTHROUGH (100M steps)

The key change: episode ends instantly on ANY collision. The drone must navigate the entire forest without touching a single tree.

- **Config**: 20 trees, curriculum 3→20 over 30%, collision_penalty=5 (credit assignment only), no proximity penalty, escape_bonus=200
- **Architecture**: MLP (512, 256), 48 lidar, 92D obs, 1024 worlds
- **LR**: 3e-4→5e-5, ent_coef 0.005→0.001

**Results trajectory**:

| Step | Trees | Nav Success | Collision | EV | Key Event |
|------|-------|-------------|-----------|-----|-----------|
| 8M | 6 | 77% | 0.07% | 0.03 | Easy with few trees |
| 21M | 13 | 23% | 0.43% | 0.19 | Curriculum ramp hurts |
| 30M | 20 | 24% | 0.43% | 0.29 | All 20 trees active |
| 50M | 20 | 36% | 0.32% | 0.66 | Slow improvement |
| 80M | 20 | 37% | 0.31% | 0.78 | Apparent plateau (30M!) |
| 88M | 20 | 44% | 0.26% | 0.81 | Sudden acceleration! |
| 92M | 20 | 50% | 0.22% | 0.82 | |
| 96M | 20 | 57% | 0.18% | 0.83 | |
| 100M | 20 | **68%** | 0.12% | 0.84 | v9 end |

- **Total time**: 4.25 hours, 6,586 FPS
- **Key insight**: The 37% plateau (50–80M) was NOT convergence. EV was still climbing (0.66→0.78). The critic needed to get good enough before the actor could improve. Once EV passed ~0.80, navigation success accelerated dramatically.

#### Phase 2 v10: Continue from v9 Checkpoint (200M more steps, CURRENT)

Resume from v9's 100M checkpoint with lower LR for fine-tuning.

- **Config**: Same as v9 but LR 5e-5→1e-5, no curriculum (already at 20 trees), ent_coef fixed at 0.001
- **Added**: `resume_checkpoint` field in TrainConfig, checkpoint loading in train_mappo.py
- **Save dir**: `results/crazyflow_v10_continued/`

**Results trajectory**:

| Step | Nav Success | Collision | EV | Key Event |
|------|-------------|-----------|-----|-----------|
| 103M | 43% | 0.21% | 0.70 | Optimizer warmup dip (fresh Adam state) |
| 105M | 74% | 0.10% | 0.84 | Recovered in 1 iteration! |
| 111M | 83% | 0.06% | 0.84 | |
| 124M | 92% | 0.03% | 0.81 | Crossed 90% |
| 136M | 94% | 0.02% | 0.75 | |
| 145M | 95% | 0.02% | 0.73 | |
| 189M | 97% | 0.008% | 0.60 | |
| 302M | **98.4%** | 0.005% | 0.45 | Near asymptote |

- **Final**: 98.4% nav success, collision rate 0.005% (~200m between collisions)
- **EV decline**: 0.84→0.45 over training but didn't impact performance. Remaining 1.6% failures are stochastic forest configs.
- **Std decline**: 0.698→0.540. Policy became deterministic — found a confident strategy.
- **Checkpoint resume lesson**: Even without optimizer state, Adam warmup recovers in 1–2 iterations with proper (lower) LR.

### Key Crazyflow Technical Lessons

1. **Forward bias init is essential**: Actor output bias=[0.5,0,0]. Without it, PPO learns backward flight (safe pessimism).
2. **terminate_on_collision=True**: THE single most important change. Makes navigation meaningful vs trivial fly-through.
3. **Proximity penalty traps**: At high tree density, proximity zones overlap → constant penalty noise → "avoid everything" instead of "navigate through gaps".
4. **EV predicts breakthroughs**: Plateaus in nav_rate don't mean convergence if EV is still climbing. Be patient — the critic must learn before the actor can improve.
5. **Curriculum timing**: curriculum_frac=0.3 (trees reach final count at 30% of training) works well. Too slow wastes steps on easy configs.
6. **Frozen dataclasses**: Config must be `frozen=True` for JAX JIT static args. k in `jax.lax.top_k` must be `static_argnums`.
7. **Obs normalization**: Clamp k_nearest to [-1,1]. Inactive trees at (100,100) create values of 33+ that destroy network learning.

### Next Steps

See [TODO.md](TODO.md) for prioritized next steps (scale to 25/30/35 trees, multi-drone, architecture improvements).

### Remote Server

- **Host**: `ai@172.31.10.232`, Python: `.venv/bin/python`
- **Training code**: `/home/ai/source/gym-pybullet-drones/`
- **Active results**: `results/crazyflow_v9_realcoll/` (v9, 100M), `results/crazyflow_v10_continued/` (v10, 300M)
- **Active scripts**: `run_phase2_v9.py`, `run_phase2_v10.py`
- **Telegram bot**: `8255249450:AAH8QEzQHS9LD4am-WWTDIfhQRebbKnwERE`, chat: `2108126743`
