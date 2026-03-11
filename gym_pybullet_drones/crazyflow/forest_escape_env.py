"""ForestEscape environment using Crazyflow JAX simulator.

Wraps crazyflow.Sim with forest generation, collision detection,
synthetic lidar observations, and shaped rewards. All operations
are vectorized over n_worlds for GPU training.
"""

import jax
import jax.numpy as jnp
import functools
from flax import struct

from .config import ForestConfig, ObsConfig, RewardConfig, TrainConfig
from .obs_utils import lidar_scan, k_nearest_trees, compute_tree_distances
from .reward import compute_reward


@struct.dataclass
class ForestState:
    """Mutable environment state carried between steps."""
    tree_xy: jnp.ndarray          # (n_worlds, n_trees, 2)
    prev_min_x: jnp.ndarray      # (n_worlds,)
    prev_drone_x: jnp.ndarray    # (n_worlds, n_drones) per-drone x for progress
    milestone_flags: jnp.ndarray  # (n_worlds, n_drones, 4)
    step_count: jnp.ndarray       # (n_worlds,) int
    done: jnp.ndarray             # (n_worlds,) bool


def generate_forest(rng_key, n_worlds, fc: ForestConfig, drone_start_xy=None,
                    n_active_trees=None):
    """Generate grid+jitter tree positions for all worlds.

    Args:
        rng_key: JAX PRNG key
        n_worlds: number of parallel worlds
        fc: ForestConfig
        drone_start_xy: (n_drones, 2) start positions to protect
        n_active_trees: int, how many trees to activate (curriculum).
                        Trees beyond this are moved to (100, 100).
                        If None, uses fc.n_trees.

    Returns:
        tree_xy: (n_worlds, n_trees, 2)
    """
    if n_active_trees is None:
        n_active_trees = fc.n_trees

    # Grid layout (always based on max n_trees for consistent spacing)
    x_range = fc.forest_x_range[1] - fc.forest_x_range[0]  # 5.0
    y_range = fc.forest_y_range[1] - fc.forest_y_range[0]  # 4.0
    area = x_range * y_range
    spacing = jnp.sqrt(area / fc.n_trees)

    nx = int(x_range / spacing) + 1
    ny = int(y_range / spacing) + 1

    # Create grid centers
    xs = jnp.linspace(fc.forest_x_range[0], fc.forest_x_range[1], nx)
    ys = jnp.linspace(fc.forest_y_range[0], fc.forest_y_range[1], ny)
    grid_x, grid_y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (nx*ny, 2)

    n_grid = grid_points.shape[0]

    key1, key2 = jax.random.split(rng_key)

    # Use first n_trees grid points
    indices = jnp.arange(min(n_grid, fc.n_trees))
    base_xy = grid_points[indices]  # (n_trees, 2)

    # Broadcast to all worlds and add jitter
    base_xy = jnp.broadcast_to(base_xy, (n_worlds, fc.n_trees, 2))
    jitter = jax.random.uniform(
        key1, (n_worlds, fc.n_trees, 2), minval=-0.2, maxval=0.2
    )
    tree_xy = base_xy + jitter

    # Curriculum: deactivate trees beyond n_active_trees
    tree_indices = jnp.arange(fc.n_trees)
    inactive_mask = tree_indices >= n_active_trees  # (n_trees,)
    far_away = jnp.array([100.0, 100.0])
    tree_xy = jnp.where(
        inactive_mask[None, :, None],  # (1, T, 1)
        far_away[None, None, :],
        tree_xy,
    )

    # Remove trees near drone starts (protected zone)
    if drone_start_xy is not None:
        diff = tree_xy[:, :, None, :] - drone_start_xy[None, None, :, :]  # (W, T, D, 2)
        dists = jnp.linalg.norm(diff, axis=-1)  # (W, T, D)
        too_close = (dists < fc.protected_radius).any(axis=-1)  # (W, T)
        tree_xy = jnp.where(too_close[:, :, None], far_away[None, None, :], tree_xy)

    return tree_xy


def make_drone_starts(n_drones):
    """Default starting positions for drones.

    Returns:
        positions: (n_drones, 3) - [x, y, z]
    """
    if n_drones == 1:
        return jnp.array([[-2.5, 0.0, 0.8]])
    elif n_drones == 3:
        return jnp.array([
            [-2.5, -0.3, 0.8],
            [-2.5, 0.0, 0.8],
            [-2.5, 0.3, 0.8],
        ])
    else:
        # Evenly space drones along y-axis
        ys = jnp.linspace(-0.3, 0.3, n_drones)
        return jnp.stack([
            jnp.full(n_drones, -2.5),
            ys,
            jnp.full(n_drones, 0.8),
        ], axis=-1)


def quat_to_rpy(quat):
    """Convert quaternion [x,y,z,w] to roll-pitch-yaw.

    Args:
        quat: (..., 4) quaternion array (x,y,z,w convention)
    Returns:
        rpy: (..., 3) roll-pitch-yaw in radians
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr, cosr)

    # Pitch
    sinp = 2.0 * (w * y - z * x)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # Yaw
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny, cosy)

    return jnp.stack([roll, pitch, yaw], axis=-1)


@functools.partial(jax.jit, static_argnums=(3, 4))
def compute_obs(sim_states, forest_state, step_count,
                oc: ObsConfig = ObsConfig(),
                fc: ForestConfig = ForestConfig(),
                episode_len: int = 1500):
    """Build observation vector for each drone.

    Args:
        sim_states: dict with pos, quat, vel, ang_vel arrays
                    each shaped (n_worlds, n_drones, ...)
        forest_state: ForestState with tree positions
        step_count: (n_worlds,) current step
        oc: ObsConfig
        fc: ForestConfig
        episode_len: max episode length

    Returns:
        obs: (n_worlds, n_drones, obs_dim)
    """
    pos = sim_states["pos"]        # (W, D, 3)
    quat = sim_states["quat"]      # (W, D, 4)
    vel = sim_states["vel"]        # (W, D, 3)
    ang_vel = sim_states["ang_vel"]  # (W, D, 3)

    n_worlds, n_drones, _ = pos.shape

    # 1. Kinematic state: pos(3) + rpy(3) + vel(3) + ang_vel(3) = 12D
    # Normalize each component to roughly [-1, 1]
    rpy = quat_to_rpy(quat)  # (W, D, 3)
    pos_norm = pos / 3.0             # pos range ~[-3, 3] -> [-1, 1]
    rpy_norm = rpy / jnp.pi         # rpy range [-pi, pi] -> [-1, 1]
    vel_norm = vel / 1.0             # vel range ~[-1, 1] (already OK)
    ang_vel_norm = ang_vel / 5.0     # ang_vel range ~[-5, 5] -> [-1, 1]
    kin = jnp.concatenate([pos_norm, rpy_norm, vel_norm, ang_vel_norm], axis=-1)  # (W, D, 12)

    # 2. Relative drone positions: 6D (to other 2 drones, zeros if <3)
    if n_drones >= 3:
        rel_0 = (jnp.roll(pos, -1, axis=1) - pos) / 3.0  # normalized
        rel_1 = (jnp.roll(pos, -2, axis=1) - pos) / 3.0
        rel_drones = jnp.concatenate([rel_0, rel_1], axis=-1)  # (W, D, 6)
    else:
        rel_drones = jnp.zeros((n_worlds, n_drones, 6))

    # 3. Goal progress + time remaining: 2D
    x_norm = (pos[:, :, 0] - (-3.0)) / (fc.escape_threshold - (-3.0))  # [0, 1]
    x_norm = jnp.clip(x_norm, 0.0, 1.0)
    time_remaining = 1.0 - step_count[:, None] / episode_len  # (W, 1)
    time_remaining = jnp.broadcast_to(time_remaining, (n_worlds, n_drones))
    goal_time = jnp.stack([x_norm, time_remaining], axis=-1)  # (W, D, 2)

    # 4. Lidar: 24D
    drone_xy = pos[:, :, :2]  # (W, D, 2)
    yaw = rpy[:, :, 2]  # (W, D)
    lidar = lidar_scan(
        drone_xy, yaw, forest_state.tree_xy, fc.tree_radius,
        n_rays=oc.n_lidar_rays, fov_deg=oc.lidar_fov_deg,
        max_range=oc.lidar_max_range,
    )  # (W, D, 24)

    # 5. K-nearest trees: 16D (8 trees * 2 coords), normalized by max range
    nearest = k_nearest_trees(
        drone_xy, forest_state.tree_xy, k=oc.k_nearest_trees,
    ) / oc.lidar_max_range
    nearest = jnp.clip(nearest, -1.0, 1.0)  # Clamp: inactive trees at (100,100) would be 33+

    # Concatenate all: 12 + 6 + 2 + 24 + 16 = 60D
    obs = jnp.concatenate([kin, rel_drones, goal_time, lidar, nearest], axis=-1)
    return obs


class ForestEscapeEnv:
    """ForestEscape environment wrapping Crazyflow Sim.

    Not a Gymnasium env — designed for direct use in a JIT-compiled
    MAPPO training loop. All methods return JAX arrays.
    """

    def __init__(self, tc: TrainConfig = TrainConfig(),
                 fc: ForestConfig = ForestConfig(),
                 oc: ObsConfig = ObsConfig(),
                 rc: RewardConfig = RewardConfig()):
        self.tc = tc
        self.fc = fc
        self.oc = oc
        self.rc = rc
        self.n_worlds = tc.n_worlds
        self.n_drones = tc.n_drones
        self.episode_len = tc.episode_len_steps
        self.drone_starts = make_drone_starts(tc.n_drones)

        # Tree curriculum state
        self.n_active_trees = fc.n_trees_start

        # Crazyflow Sim will be created lazily
        self.sim = None

    def _create_sim(self):
        """Create the Crazyflow Sim instance."""
        from crazyflow.sim import Sim, Physics
        from crazyflow.control import Control

        self.sim = Sim(
            n_worlds=self.n_worlds,
            n_drones=self.n_drones,
            physics=Physics.analytical,
            control=Control.state,
            freq=self.tc.physics_freq,
            state_freq=self.tc.ctrl_freq,
            device="gpu",
        )

    def _get_sim_states(self):
        """Extract state arrays from Crazyflow sim."""
        return {
            "pos": self.sim.data.states.pos,
            "quat": self.sim.data.states.quat,
            "vel": self.sim.data.states.vel,
            "ang_vel": self.sim.data.states.ang_vel,
        }

    def set_curriculum(self, progress_frac):
        """Update curriculum based on training progress.

        Args:
            progress_frac: float in [0, 1], fraction of total training done
        """
        fc = self.fc
        # Linear ramp from n_trees_start to n_trees_final over curriculum_frac
        if progress_frac >= fc.curriculum_frac:
            self.n_active_trees = fc.n_trees_final
        else:
            t = progress_frac / fc.curriculum_frac  # [0, 1] within curriculum phase
            n = fc.n_trees_start + t * (fc.n_trees_final - fc.n_trees_start)
            self.n_active_trees = int(n)

    def reset(self, rng_key):
        """Reset all worlds.

        Args:
            rng_key: JAX PRNG key

        Returns:
            obs: (n_worlds, n_drones, obs_dim)
            forest_state: ForestState
        """
        if self.sim is None:
            self._create_sim()

        key1, key2 = jax.random.split(rng_key)

        # Generate forest with curriculum tree count
        tree_xy = generate_forest(
            key1, self.n_worlds, self.fc,
            drone_start_xy=self.drone_starts[:, :2],
            n_active_trees=self.n_active_trees,
        )

        # Reset Crazyflow sim
        self.sim.reset()

        # Set drone starting positions
        start_pos = jnp.broadcast_to(
            self.drone_starts[None, :, :],
            (self.n_worlds, self.n_drones, 3),
        )
        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(pos=start_pos)
        )

        # Create forest state
        forest_state = ForestState(
            tree_xy=tree_xy,
            prev_min_x=jnp.full(self.n_worlds, -2.5),
            prev_drone_x=jnp.full((self.n_worlds, self.n_drones), -2.5),
            milestone_flags=jnp.zeros((self.n_worlds, self.n_drones, 4), dtype=bool),
            step_count=jnp.zeros(self.n_worlds, dtype=jnp.int32),
            done=jnp.zeros(self.n_worlds, dtype=bool),
        )

        # Compute initial obs
        sim_states = self._get_sim_states()
        obs = compute_obs(sim_states, forest_state, forest_state.step_count,
                          self.oc, self.fc, self.episode_len)

        return obs, forest_state

    def step(self, actions, forest_state):
        """Take one environment step.

        Args:
            actions: (n_worlds, n_drones, 3) in [-1, 1]
            forest_state: current ForestState

        Returns:
            obs: (n_worlds, n_drones, obs_dim)
            rewards: (n_worlds, n_drones)
            dones: (n_worlds,) bool
            forest_state: updated ForestState
            info: dict of metrics
        """
        # Apply actions: target_pos = current_pos + action_scale * action
        current_pos = self.sim.data.states.pos  # (W, D, 3)
        target_pos = current_pos + self.tc.action_scale * actions

        # Clamp height to reasonable bounds
        target_z = jnp.clip(target_pos[:, :, 2], 0.1, 2.0)
        target_pos = target_pos.at[:, :, 2].set(target_z)

        # Build state control command (13D: pos + vel + acc + yaw + rates)
        cmd = jnp.zeros((self.n_worlds, self.n_drones, 13))
        cmd = cmd.at[:, :, :3].set(target_pos)

        # Step physics
        self.sim.state_control(cmd)
        self.sim.step(self.tc.n_substeps)

        # Get updated state
        sim_states = self._get_sim_states()
        pos = sim_states["pos"]
        vel = sim_states["vel"]

        # Collision detection
        drone_xy = pos[:, :, :2]
        _, in_collision, _, proximity_depth = compute_tree_distances(
            drone_xy, forest_state.tree_xy, self.fc.tree_radius,
            self.fc.collision_dist, self.fc.proximity_dist,
        )

        # Compute reward
        step_count = forest_state.step_count + 1
        per_drone_reward, team_reward, new_min_x, new_milestones, info = compute_reward(
            pos, vel,
            forest_state.prev_min_x,
            forest_state.prev_drone_x,
            forest_state.milestone_flags,
            in_collision, proximity_depth,
            step_count,
            self.rc, self.fc, self.episode_len,
        )

        # Check termination: all escaped, time up, or collision
        all_escaped = (pos[:, :, 0] > self.fc.escape_threshold).all(axis=1)
        time_up = step_count >= self.episode_len
        any_collision = in_collision.any(axis=1)  # (n_worlds,)
        dones = all_escaped | time_up
        if self.fc.terminate_on_collision:
            dones = dones | any_collision

        # Update forest state
        new_forest_state = ForestState(
            tree_xy=forest_state.tree_xy,
            prev_min_x=new_min_x,
            prev_drone_x=pos[:, :, 0],  # Current x becomes prev for next step
            milestone_flags=new_milestones,
            step_count=step_count,
            done=dones,
        )

        # Compute obs
        obs = compute_obs(sim_states, new_forest_state, step_count,
                          self.oc, self.fc, self.episode_len)

        return obs, per_drone_reward, dones, new_forest_state, info

    def reset_done_worlds(self, forest_state, rng_key):
        """Reset worlds where done=True.

        Args:
            forest_state: current ForestState
            rng_key: JAX PRNG key

        Returns:
            obs: (n_worlds, n_drones, obs_dim)
            forest_state: updated ForestState (done worlds reset)
        """
        done_mask = forest_state.done

        # Generate new trees for done worlds (with curriculum)
        key1, key2 = jax.random.split(rng_key)
        new_trees = generate_forest(
            key1, self.n_worlds, self.fc,
            drone_start_xy=self.drone_starts[:, :2],
            n_active_trees=self.n_active_trees,
        )

        # Merge: keep old trees for non-done worlds, new trees for done worlds
        tree_xy = jnp.where(
            done_mask[:, None, None],
            new_trees,
            forest_state.tree_xy,
        )

        # Reset positions for done worlds
        start_pos = jnp.broadcast_to(
            self.drone_starts[None, :, :],
            (self.n_worlds, self.n_drones, 3),
        )
        current_pos = self.sim.data.states.pos
        new_pos = jnp.where(done_mask[:, None, None], start_pos, current_pos)

        # Reset velocities for done worlds
        current_vel = self.sim.data.states.vel
        new_vel = jnp.where(done_mask[:, None, None], 0.0, current_vel)

        current_ang_vel = self.sim.data.states.ang_vel
        new_ang_vel = jnp.where(done_mask[:, None, None], 0.0, current_ang_vel)

        # Update sim state
        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(
                pos=new_pos,
                vel=new_vel,
                ang_vel=new_ang_vel,
            )
        )

        # Reset forest state for done worlds
        new_forest_state = ForestState(
            tree_xy=tree_xy,
            prev_min_x=jnp.where(done_mask, -2.5, forest_state.prev_min_x),
            prev_drone_x=jnp.where(
                done_mask[:, None],
                jnp.full_like(forest_state.prev_drone_x, -2.5),
                forest_state.prev_drone_x,
            ),
            milestone_flags=jnp.where(
                done_mask[:, None, None],
                jnp.zeros_like(forest_state.milestone_flags),
                forest_state.milestone_flags,
            ),
            step_count=jnp.where(done_mask, 0, forest_state.step_count),
            done=jnp.zeros(self.n_worlds, dtype=bool),
        )

        # Compute obs
        sim_states = self._get_sim_states()
        obs = compute_obs(sim_states, new_forest_state, new_forest_state.step_count,
                          self.oc, self.fc, self.episode_len)

        return obs, new_forest_state
