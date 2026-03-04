"""DenseForestAviary: Multi-agent cooperative navigation through a dense forest.

3 drones share their camera vision via a cross-attention communication
mechanism (MAVSAC policy) to navigate through procedurally generated
forests of cylindrical tree obstacles toward sequential waypoints.

Each drone captures an onboard RGB image. The MAVSAC policy fuses these
visual observations across agents using cross-attention, enabling
collective perception — a lead drone can alert flanking drones about
upcoming obstacles, and lateral drones provide peripheral coverage.

Observation Space (Dict):
    "vision": (NUM_DRONES, IMG_H, IMG_W, 4) - RGBA images per drone
    "state":  (NUM_DRONES, state_dim) - kinematic + neighbor + waypoint + obstacle info

Example
-------
    >>> env = DenseForestAviary(gui=True)
    >>> obs, info = env.reset()
    >>> obs["vision"].shape  # (3, 48, 64, 4)
    >>> obs["state"].shape   # (3, state_dim)

"""
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType


class DenseForestAviary(BaseRLAviary):
    """Multi-agent RL: 3 drones sharing vision to navigate a dense forest."""

    NUM_AGENTS = 3
    NUM_NEAREST_OBSTACLES = 3
    # State obs per drone: 12 kin + 6 neighbor_rel_pos + 3 wp_dir + 1 wp_dist + 9 obstacles + 1 time = 32
    STATE_OBS_DIM = 32

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 3,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,  # Must be compatible with 24fps RGB capture
                 gui=False,
                 record=False,
                 act: ActionType = ActionType.RPM,
                 num_trees: int = 35,
                 forest_x_range: tuple = (-3.0, 3.0),
                 forest_y_range: tuple = (-2.0, 2.0),
                 tree_radius: float = 0.06,
                 tree_height: float = 1.8,
                 ):
        """Initialize the DenseForestAviary.

        Always uses RGB observation (shared vision) combined with kinematic state.

        Parameters
        ----------
        num_trees : int
            Number of cylindrical tree obstacles.
        forest_x_range : tuple
            (min_x, max_x) of the forest region.
        forest_y_range : tuple
            (min_y, max_y) of the forest region.
        tree_radius : float
            Radius of each tree trunk cylinder (meters).
        tree_height : float
            Height of each tree trunk cylinder (meters).
        """
        self.EPISODE_LEN_SEC = 30
        self.NUM_TREES = num_trees
        self.FOREST_X_RANGE = forest_x_range
        self.FOREST_Y_RANGE = forest_y_range
        self.TREE_RADIUS = tree_radius
        self.TREE_HEIGHT = tree_height

        # Obstacle tracking
        self.tree_positions = []
        self.tree_ids = []

        # Waypoints: zigzag path through the forest
        self.WAYPOINTS = np.array([
            [-1.5, -0.5, 1.0],
            [-0.5,  0.8, 1.0],
            [ 0.5, -0.5, 1.0],
            [ 1.5,  0.8, 1.0],
            [ 2.5,  0.0, 1.0],
        ])
        self.WAYPOINT_REACH_DIST = 0.35
        self.current_waypoint_idx = 0  # Team waypoint (shared)
        self.waypoint_marker_ids = []

        # Formation parameters
        self.FORMATION_MIN_DIST = 0.25
        self.FORMATION_MAX_DIST = 1.5
        self.COLLISION_DIST = 0.15  # tree collision threshold

        # Initial formation: triangle at forest entrance
        if initial_xyzs is None:
            initial_xyzs = np.array([
                [-2.5, -0.3, 0.8],
                [-2.5,  0.0, 1.0],
                [-2.5,  0.3, 0.8],
            ])

        # Force RGB observation for shared vision
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=ObservationType.RGB,  # Always use RGB for vision
                         act=act
                         )

    ################################################################################

    def reset(self, seed=None, options=None):
        """Reset environment with new forest layout."""
        # Remove old trees
        for tree_id in self.tree_ids:
            p.removeBody(tree_id, physicsClientId=self.CLIENT)
        for marker_id in self.waypoint_marker_ids:
            p.removeBody(marker_id, physicsClientId=self.CLIENT)

        self.tree_ids = []
        self.tree_positions = []
        self.waypoint_marker_ids = []
        self.current_waypoint_idx = 0

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    ################################################################################

    def _addObstacles(self):
        """Generate procedural forest of cylindrical trees and waypoint markers."""
        self.tree_positions = []
        self.tree_ids = []
        self.waypoint_marker_ids = []

        # Protected zones: no trees near drone starts, waypoints
        protected_positions = []
        if self.INIT_XYZS is not None:
            for pos in self.INIT_XYZS:
                protected_positions.append(pos[:2])
        for wp in self.WAYPOINTS:
            protected_positions.append(wp[:2])
        protected_radius = 0.45

        # Generate tree positions via rejection sampling
        np.random.seed(None)
        attempts = 0
        max_attempts = self.NUM_TREES * 20

        while len(self.tree_positions) < self.NUM_TREES and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(self.FOREST_X_RANGE[0], self.FOREST_X_RANGE[1])
            y = np.random.uniform(self.FOREST_Y_RANGE[0], self.FOREST_Y_RANGE[1])
            pos_2d = np.array([x, y])

            too_close = False
            for prot in protected_positions:
                if np.linalg.norm(pos_2d - np.array(prot)) < protected_radius:
                    too_close = True
                    break

            if not too_close:
                for existing in self.tree_positions:
                    if np.linalg.norm(pos_2d - np.array(existing[:2])) < self.TREE_RADIUS * 4:
                        too_close = True
                        break

            if not too_close:
                pos_3d = [x, y, self.TREE_HEIGHT / 2]
                self.tree_positions.append(pos_3d)

                col_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=self.TREE_RADIUS,
                    height=self.TREE_HEIGHT,
                    physicsClientId=self.CLIENT
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=self.TREE_RADIUS,
                    length=self.TREE_HEIGHT,
                    rgbaColor=[0.35, 0.22, 0.08, 1.0],
                    physicsClientId=self.CLIENT
                )
                tree_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_shape,
                    baseVisualShapeIndex=vis_shape,
                    basePosition=pos_3d,
                    physicsClientId=self.CLIENT
                )
                self.tree_ids.append(tree_id)

                # Green canopy
                canopy_vis = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=self.TREE_RADIUS * 3,
                    rgbaColor=[0.1, 0.5, 0.1, 0.7],
                    physicsClientId=self.CLIENT
                )
                canopy_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=canopy_vis,
                    basePosition=[x, y, self.TREE_HEIGHT + self.TREE_RADIUS],
                    physicsClientId=self.CLIENT
                )
                self.tree_ids.append(canopy_id)

        # Waypoint markers (semi-transparent green spheres)
        for wp in self.WAYPOINTS:
            wp_vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.12,
                rgbaColor=[0.0, 1.0, 0.2, 0.4],
                physicsClientId=self.CLIENT
            )
            wp_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=wp_vis,
                basePosition=wp.tolist(),
                physicsClientId=self.CLIENT
            )
            self.waypoint_marker_ids.append(wp_id)

        # Forest-colored ground
        p.changeVisualShape(
            p.loadURDF("plane.urdf", physicsClientId=self.CLIENT),
            -1,
            rgbaColor=[0.2, 0.35, 0.1, 1.0],
            physicsClientId=self.CLIENT
        )

    ################################################################################

    def _observationSpace(self):
        """Dict observation space: RGB vision + kinematic state per drone.

        Returns
        -------
        spaces.Dict
            "vision": Box(0, 255, shape=(NUM_DRONES, H, W, 4), uint8)
            "state":  Box(-inf, inf, shape=(NUM_DRONES, STATE_OBS_DIM), float32)
        """
        return spaces.Dict({
            "vision": spaces.Box(
                low=0, high=255,
                shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4),
                dtype=np.uint8
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.NUM_DRONES, self.STATE_OBS_DIM),
                dtype=np.float32
            ),
        })

    ################################################################################

    def _computeObs(self):
        """Compute Dict observation: RGB images + extended state per drone.

        Returns
        -------
        dict
            "vision": ndarray (NUM_DRONES, H, W, 4) - RGBA camera images
            "state":  ndarray (NUM_DRONES, STATE_OBS_DIM) - extended kinematic state
        """
        # --- Capture RGB images from each drone's onboard camera ---
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
            for i in range(self.NUM_DRONES):
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(
                    i, segmentation=False
                )
                if self.RECORD:
                    self._exportImage(
                        img_type=ImageType.RGB,
                        img_input=self.rgb[i],
                        path=self.ONBOARD_IMG_PATH + "drone_" + str(i),
                        frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                    )

        vision = np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype(np.uint8)

        # --- Compute extended kinematic state per drone ---
        state_obs = np.zeros((self.NUM_DRONES, self.STATE_OBS_DIM), dtype=np.float32)
        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]

        for i in range(self.NUM_DRONES):
            pos_i = states[i][0:3]

            # Own kinematic: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz] (12D)
            state_obs[i, 0:3] = states[i][0:3]       # position
            state_obs[i, 3:6] = states[i][7:10]       # rpy
            state_obs[i, 6:9] = states[i][10:13]      # linear velocity
            state_obs[i, 9:12] = states[i][13:16]     # angular velocity

            # Relative positions of other 2 drones (6D)
            others = [j for j in range(self.NUM_DRONES) if j != i]
            for k, j in enumerate(others):
                state_obs[i, 12 + k * 3:12 + (k + 1) * 3] = states[j][0:3] - pos_i

            # Next waypoint relative position (3D) + distance (1D)
            if self.current_waypoint_idx < len(self.WAYPOINTS):
                wp = self.WAYPOINTS[self.current_waypoint_idx]
                rel = wp - pos_i
                dist = np.linalg.norm(rel)
                state_obs[i, 18:21] = rel
                state_obs[i, 21] = dist

            # Nearest 3 obstacles relative positions (9D)
            if len(self.tree_positions) > 0:
                tree_arr = np.array(self.tree_positions)
                dists = np.linalg.norm(tree_arr - pos_i, axis=1)
                nearest_idx = np.argsort(dists)[:self.NUM_NEAREST_OBSTACLES]
                for k, idx in enumerate(nearest_idx):
                    state_obs[i, 22 + k * 3:22 + (k + 1) * 3] = tree_arr[idx] - pos_i

            # Normalized time remaining (1D)
            time_elapsed = self.step_counter / (self.EPISODE_LEN_SEC * self.PYB_FREQ)
            state_obs[i, 31] = max(0.0, 1.0 - time_elapsed)

        return {"vision": vision, "state": state_obs}

    ################################################################################

    def _computeReward(self):
        """Cooperative reward: waypoint progress + collision avoidance + formation."""
        reward = 0.0

        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        positions = np.array([s[0:3] for s in states])
        velocities = np.array([s[10:13] for s in states])

        # --- Team centroid for waypoint tracking ---
        centroid = np.mean(positions, axis=0)

        if self.current_waypoint_idx < len(self.WAYPOINTS):
            wp = self.WAYPOINTS[self.current_waypoint_idx]
            centroid_dist = np.linalg.norm(centroid - wp)

            # Distance-based shaping
            reward -= centroid_dist * 0.02

            # Velocity towards waypoint
            if centroid_dist > 0.1:
                dir_to_wp = (wp - centroid) / centroid_dist
                avg_vel = np.mean(velocities, axis=0)
                vel_towards = np.dot(avg_vel, dir_to_wp)
                reward += vel_towards * 0.15

            # Waypoint reached
            if centroid_dist < self.WAYPOINT_REACH_DIST:
                self.current_waypoint_idx += 1
                reward += 100.0
                if self.current_waypoint_idx >= len(self.WAYPOINTS):
                    reward += 200.0

        # --- Per-drone rewards ---
        for i in range(self.NUM_DRONES):
            pos_i = positions[i]

            # Obstacle proximity penalty
            if len(self.tree_positions) > 0:
                tree_arr = np.array(self.tree_positions)
                dists_2d = np.linalg.norm(tree_arr[:, :2] - pos_i[:2], axis=1)
                min_dist = np.min(dists_2d)
                if min_dist < self.COLLISION_DIST:
                    reward -= 15.0
                elif min_dist < 0.3:
                    reward -= (0.3 - min_dist) * 8.0

            # Height maintenance
            height_error = abs(pos_i[2] - 1.0)
            if height_error > 0.5:
                reward -= height_error * 0.5

            # Tilt penalty
            rpy = states[i][7:10]
            tilt = abs(rpy[0]) + abs(rpy[1])
            if tilt > 0.3:
                reward -= tilt * 0.2

        # --- Formation reward ---
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                inter_dist = np.linalg.norm(positions[i] - positions[j])
                if inter_dist < self.FORMATION_MIN_DIST:
                    reward -= 3.0
                elif inter_dist > self.FORMATION_MAX_DIST:
                    reward -= (inter_dist - self.FORMATION_MAX_DIST) * 1.0
                else:
                    reward += 0.1

        # --- Cooperative coverage bonus ---
        if self.current_waypoint_idx < len(self.WAYPOINTS):
            wp = self.WAYPOINTS[self.current_waypoint_idx]
            move_dir = wp - centroid
            move_dist = np.linalg.norm(move_dir)
            if move_dist > 0.1:
                move_dir_norm = move_dir / move_dist
                perp_norms = []
                for pos in positions:
                    rel = pos - centroid
                    perp = rel - np.dot(rel, move_dir_norm) * move_dir_norm
                    perp_norms.append(np.linalg.norm(perp))
                avg_spread = np.mean(perp_norms)
                if 0.2 < avg_spread < 0.8:
                    reward += avg_spread * 0.3

        reward -= 0.05
        return reward

    ################################################################################

    def _computeTerminated(self):
        """Episode terminates when all waypoints are reached."""
        return self.current_waypoint_idx >= len(self.WAYPOINTS)

    ################################################################################

    def _computeTruncated(self):
        """Truncate on out-of-bounds, excessive tilt, or timeout."""
        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]

        for i in range(self.NUM_DRONES):
            pos = states[i][0:3]
            rpy = states[i][7:10]

            if (abs(pos[0]) > 4.0 or abs(pos[1]) > 3.0 or
                    pos[2] > 2.5 or pos[2] < 0.05):
                return True
            if abs(rpy[0]) > 1.2 or abs(rpy[1]) > 1.2:
                return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    ################################################################################

    def _computeInfo(self):
        """Return info dict with progress metrics."""
        positions = np.array([self._getDroneStateVector(i)[0:3]
                              for i in range(self.NUM_DRONES)])
        centroid = np.mean(positions, axis=0)

        inter_dists = []
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                inter_dists.append(np.linalg.norm(positions[i] - positions[j]))

        min_obs_dist = float('inf')
        if len(self.tree_positions) > 0:
            tree_arr = np.array(self.tree_positions)
            for i in range(self.NUM_DRONES):
                dists_2d = np.linalg.norm(tree_arr[:, :2] - positions[i, :2], axis=1)
                min_obs_dist = min(min_obs_dist, np.min(dists_2d))

        return {
            "waypoints_reached": self.current_waypoint_idx,
            "total_waypoints": len(self.WAYPOINTS),
            "centroid": centroid.tolist(),
            "mean_inter_drone_dist": np.mean(inter_dists) if inter_dists else 0.0,
            "min_obstacle_dist": min_obs_dist,
            "all_waypoints_reached": self.current_waypoint_idx >= len(self.WAYPOINTS),
        }
