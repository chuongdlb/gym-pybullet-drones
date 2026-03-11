"""Observation utilities: synthetic lidar and k-nearest obstacles (pure JAX)."""

import jax
import jax.numpy as jnp
import functools


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def lidar_scan(drone_xy, drone_yaw, tree_xy, tree_radius,
               n_rays=24, fov_deg=180.0, max_range=3.0):
    """Compute synthetic lidar distances via ray-circle intersection.

    All inputs/outputs are batched over (n_worlds, n_drones).

    Args:
        drone_xy: (n_worlds, n_drones, 2) drone positions in XY plane
        drone_yaw: (n_worlds, n_drones) drone heading angles (radians)
        tree_xy: (n_worlds, n_trees, 2) tree center positions
        tree_radius: scalar, tree trunk radius
        n_rays: number of lidar rays
        fov_deg: field of view in degrees (centered on heading)
        max_range: maximum detection range

    Returns:
        distances: (n_worlds, n_drones, n_rays) normalized [0, 1]
                   0 = at tree surface, 1 = max range (no detection)
    """
    # Ray angles relative to drone heading
    half_fov = fov_deg / 2.0 * jnp.pi / 180.0
    ray_offsets = jnp.linspace(-half_fov, half_fov, n_rays)

    # Absolute ray angles: (n_worlds, n_drones, n_rays)
    ray_angles = drone_yaw[..., None] + ray_offsets[None, None, :]

    # Ray directions: (n_worlds, n_drones, n_rays, 2)
    ray_dirs = jnp.stack([jnp.cos(ray_angles), jnp.sin(ray_angles)], axis=-1)

    # Vector from drone to each tree: (n_worlds, n_drones, n_trees, 2)
    # drone_xy: (W, D, 2) -> (W, D, 1, 2)
    # tree_xy:  (W, T, 2) -> (W, 1, T, 2)
    to_tree = tree_xy[:, None, :, :] - drone_xy[:, :, None, :]

    # For each (drone, ray, tree) triple, compute ray-circle intersection.
    # Project tree center onto ray: d = dot(to_tree, ray_dir)
    # Perpendicular distance: h² = |to_tree|² - d²
    # Hit if h < tree_radius and d > 0
    # Hit distance: d - sqrt(tree_radius² - h²)

    # to_tree: (W, D, T, 2), ray_dirs: (W, D, R, 2)
    # We need (W, D, R, T) dot products
    # Expand: to_tree -> (W, D, 1, T, 2), ray_dirs -> (W, D, R, 1, 2)
    to_tree_exp = to_tree[:, :, None, :, :]   # (W, D, 1, T, 2)
    ray_dirs_exp = ray_dirs[:, :, :, None, :]  # (W, D, R, 1, 2)

    # Projection along ray: (W, D, R, T)
    d_proj = jnp.sum(to_tree_exp * ray_dirs_exp, axis=-1)

    # Squared distance from tree center to ray: (W, D, R, T)
    to_tree_sq = jnp.sum(to_tree_exp ** 2, axis=-1)  # (W, D, 1, T)
    h_sq = to_tree_sq - d_proj ** 2
    h_sq = jnp.maximum(h_sq, 0.0)  # Numerical safety

    # Check hit conditions
    radius_sq = tree_radius ** 2
    is_hit = (h_sq < radius_sq) & (d_proj > 0)

    # Hit distance (distance along ray to circle surface)
    hit_dist = d_proj - jnp.sqrt(jnp.maximum(radius_sq - h_sq, 0.0))

    # Set non-hits to max_range
    hit_dist = jnp.where(is_hit & (hit_dist > 0), hit_dist, max_range)

    # Minimum distance across all trees for each ray: (W, D, R)
    ray_dists = jnp.min(hit_dist, axis=-1)

    # Clamp and normalize to [0, 1]
    ray_dists = jnp.clip(ray_dists, 0.0, max_range) / max_range

    return ray_dists


@functools.partial(jax.jit, static_argnums=(2,))
def k_nearest_trees(drone_xy, tree_xy, k=8):
    """Compute relative positions to K nearest trees per drone.

    Args:
        drone_xy: (n_worlds, n_drones, 2)
        tree_xy: (n_worlds, n_trees, 2)
        k: number of nearest trees

    Returns:
        rel_positions: (n_worlds, n_drones, k * 2) relative (dx, dy) flattened
    """
    # Distances: (n_worlds, n_drones, n_trees)
    diff = tree_xy[:, None, :, :] - drone_xy[:, :, None, :]  # (W, D, T, 2)
    dists = jnp.linalg.norm(diff, axis=-1)  # (W, D, T)

    # Get indices of k nearest (top_k on negated distances)
    _, indices = jax.lax.top_k(-dists, k)  # (W, D, k)

    # Gather relative positions using indices
    # diff: (W, D, T, 2), indices: (W, D, k)
    n_worlds, n_drones, _ = indices.shape

    # Advanced indexing: for each (w, d, ki), get diff[w, d, indices[w,d,ki], :]
    w_idx = jnp.arange(n_worlds)[:, None, None]  # (W, 1, 1)
    d_idx = jnp.arange(n_drones)[None, :, None]  # (1, D, 1)
    nearest_rel = diff[w_idx, d_idx, indices, :]  # (W, D, k, 2)

    # Flatten last two dims
    return nearest_rel.reshape(n_worlds, n_drones, k * 2)


@functools.partial(jax.jit, static_argnums=(3, 4))
def compute_tree_distances(drone_xy, tree_xy, tree_radius,
                           collision_dist=0.15, proximity_dist=0.45):
    """Compute per-drone minimum tree distances and collision/proximity flags.

    Args:
        drone_xy: (n_worlds, n_drones, 2)
        tree_xy: (n_worlds, n_trees, 2)
        tree_radius: scalar
        collision_dist: collision threshold (center-to-center minus radius)
        proximity_dist: proximity penalty threshold

    Returns:
        min_dist: (n_worlds, n_drones) min distance to any tree surface
        in_collision: (n_worlds, n_drones) bool
        in_proximity: (n_worlds, n_drones) bool
        proximity_depth: (n_worlds, n_drones) how deep in proximity zone
    """
    diff = drone_xy[:, :, None, :] - tree_xy[:, None, :, :]  # (W, D, T, 2)
    center_dists = jnp.linalg.norm(diff, axis=-1)  # (W, D, T)
    surface_dists = center_dists - tree_radius  # Distance to tree surface
    min_dist = jnp.min(surface_dists, axis=-1)  # (W, D)

    in_collision = min_dist < collision_dist
    in_proximity = (min_dist >= collision_dist) & (min_dist < proximity_dist)
    proximity_depth = jnp.where(in_proximity, proximity_dist - min_dist, 0.0)

    return min_dist, in_collision, in_proximity, proximity_depth
