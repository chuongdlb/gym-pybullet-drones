"""Vectorized reward function for ForestEscape (pure JAX).

Port of ForestEscapeAviary._computeReward() from v10b.
All operations batched over n_worlds. No Python control flow over array values.
"""

import jax
import jax.numpy as jnp
import functools

from .config import RewardConfig, ForestConfig


@functools.partial(jax.jit, static_argnums=(8, 9, 10))
def compute_reward(
    positions,          # (n_worlds, n_drones, 3)
    velocities,         # (n_worlds, n_drones, 3)
    prev_min_x,         # (n_worlds,)
    prev_drone_x,       # (n_worlds, n_drones) per-drone previous x (unused for now)
    milestone_flags,    # (n_worlds, n_drones, 4)
    in_collision,       # (n_worlds, n_drones) bool
    proximity_depth,    # (n_worlds, n_drones)
    step_counter,       # (n_worlds,) int
    rc: RewardConfig = RewardConfig(),
    fc: ForestConfig = ForestConfig(),
    episode_len: int = 1500,
):
    """Compute per-drone and team rewards.

    Returns:
        per_drone_reward: (n_worlds, n_drones)
        team_reward: (n_worlds,)
        new_prev_min_x: (n_worlds,)
        new_milestone_flags: (n_worlds, n_drones, 4)
        info: dict of metrics
    """
    n_drones = positions.shape[1]

    # ===== Team reward (shared across drones) =====

    # Min-x progress (slowest drone drives team reward)
    min_x = positions[:, :, 0].min(axis=1)  # (n_worlds,)
    delta_min_x = min_x - prev_min_x
    team_progress = delta_min_x * rc.progress_coef  # (n_worlds,)

    # Formation spread penalty
    x_spread = positions[:, :, 0].max(axis=1) - positions[:, :, 0].min(axis=1)
    spread_penalty = jnp.maximum(x_spread - rc.formation_spread_threshold, 0.0) * rc.formation_spread_coef

    # Time cost
    time_cost = jnp.full_like(min_x, rc.time_penalty)

    # Escape bonus (all drones past threshold)
    all_escaped = (positions[:, :, 0] > fc.escape_threshold).all(axis=1)
    escape_bonus = all_escaped.astype(jnp.float32) * rc.escape_bonus

    team_reward = team_progress - spread_penalty - time_cost + escape_bonus  # (n_worlds,)

    # ===== Per-drone shaped reward =====

    vel_x = velocities[:, :, 0]  # (n_worlds, n_drones)
    drone_x = positions[:, :, 0]  # (n_worlds, n_drones)

    # Per-drone individual forward progress (each drone rewarded for OWN movement)
    drone_delta_x = drone_x - prev_drone_x  # (n_worlds, n_drones)
    per_drone_progress = drone_delta_x * rc.progress_coef  # Same scale as team

    # Forward velocity bonus
    forward_bonus = jnp.clip(vel_x, 0.0, rc.forward_vel_max) * rc.forward_vel_bonus

    # Backward velocity penalty (proportional, very strong)
    backward = jnp.maximum(-vel_x, 0.0) * rc.backward_penalty

    # Position-based backward penalty: penalize being behind start (x < -2.5)
    behind_start = jnp.maximum(-2.5 - drone_x, 0.0) * 5.0  # (W, D)

    # Idle penalty (per-drone)
    idle = (jnp.abs(vel_x) < rc.idle_threshold).astype(jnp.float32) * rc.idle_penalty

    # Collision penalty
    collision = in_collision.astype(jnp.float32) * rc.collision_penalty

    # Proximity penalty
    proximity = proximity_depth * fc.proximity_coef

    # Speed-near-trees penalty: discourage fast flight in proximity zone
    speed_xy = jnp.sqrt(vel_x**2 + velocities[:, :, 1]**2)  # XY speed (W, D)
    speed_proximity = speed_xy * proximity_depth * rc.speed_proximity_coef

    # Height penalty
    z = positions[:, :, 2]
    height_bad = ((z < fc.height_min) | (z > fc.height_max)).astype(jnp.float32) * rc.height_penalty

    # Milestone bonuses
    milestone_thresholds = jnp.array(rc.milestones)  # (4,)
    current_milestones = drone_x[:, :, None] > milestone_thresholds[None, None, :]  # (W, D, 4)
    newly_hit = current_milestones & ~milestone_flags  # (W, D, 4)
    milestone_reward = newly_hit.sum(axis=-1).astype(jnp.float32) * rc.milestone_bonus  # (W, D)
    new_milestone_flags = milestone_flags | current_milestones

    # Sum per-drone
    per_drone_reward = (
        per_drone_progress
        + forward_bonus
        - backward
        - behind_start
        - idle
        - collision
        - proximity
        - speed_proximity
        - height_bad
        + milestone_reward
    )  # (n_worlds, n_drones)

    # Total per-drone = team share + individual shaping
    total_per_drone = team_reward[:, None] / n_drones + per_drone_reward

    # Info for logging
    info = {
        "team_progress": team_progress.mean(),
        "min_x": min_x.mean(),
        "collision_rate": in_collision.any(axis=1).mean(),
        "escape_rate": all_escaped.mean(),
        "mean_vel_x": vel_x.mean(),
    }

    return total_per_drone, team_reward, min_x, new_milestone_flags, info
