"""ForestEscapeAviary: Cooperative forest escape (v10/v10b).

Drones share camera vision via cross-attention to navigate through a
procedurally generated dense forest. All drones must reach x > ESCAPE_THRESHOLD.

v10b: supports 1-drone mode with configurable image resolution (default 128×96).
v10: min-x progress, per-drone backward penalty, milestone bonuses, formation spread.

Observation Space (Dict):
    "vision": (NUM_DRONES, IMG_H, IMG_W, 4) - RGBD images per drone (RGB + depth)
    "state":  (NUM_DRONES * 29,) - kinematics + neighbors + goal + obstacles + time

Example
-------
    >>> env = ForestEscapeAviary(gui=False, num_drones=1, img_res=(128, 96))
    >>> obs, info = env.reset()
    >>> obs["vision"].shape  # (1, 96, 128, 4)
    >>> obs["state"].shape   # (29,)
"""
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class ForestEscapeAviary(BaseRLAviary):
    """RL env: drones sharing vision to escape a dense forest (1–3 drones)."""

    NUM_AGENTS = 3
    NUM_NEAREST_OBSTACLES = 3
    # State obs per drone: 12 kin + 6 neighbor + 1 goal_progress + 9 obstacles + 1 time = 29
    STATE_OBS_DIM = 29

    ESCAPE_THRESHOLD = 2.5  # x-coordinate for "escaped"

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 3,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui=False,
                 record=False,
                 act: ActionType = ActionType.RPM,
                 num_trees: int = 35,
                 forest_x_range: tuple = (-3.0, 3.0),
                 forest_y_range: tuple = (-2.0, 2.0),
                 tree_radius: float = 0.06,
                 tree_height: float = 1.8,
                 img_res: tuple = (64, 48),
                 ):
        self.EPISODE_LEN_SEC = 30
        self.NUM_TREES = num_trees
        self.FOREST_X_RANGE = forest_x_range
        self.FOREST_Y_RANGE = forest_y_range
        self.TREE_RADIUS = tree_radius
        self.TREE_HEIGHT = tree_height

        # Obstacle tracking
        self.tree_positions = []
        self.tree_ids = []

        # Formation / collision
        self.FORMATION_MIN_DIST = 0.25
        self.FORMATION_MAX_DIST = 1.5
        self.COLLISION_DIST = 0.15

        # Escape tracking
        self._prev_min_x = None
        self._milestones = [[False] * 4 for _ in range(num_drones)]
        self.MILESTONES = [-1.5, -0.5, 0.5, 1.5]

        # Initial positions: conditional on num_drones
        if initial_xyzs is None:
            if num_drones == 1:
                initial_xyzs = np.array([[-2.5, 0.0, 1.0]])
            else:
                initial_xyzs = np.array([
                    [-2.5, -0.3, 0.8],
                    [-2.5,  0.0, 1.0],
                    [-2.5,  0.3, 0.8],
                ])

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
                         obs=ObservationType.RGB,
                         act=act)

        # Override image resolution if non-default (must be after super().__init__)
        if img_res != (64, 48):
            self.IMG_RES = np.array([img_res[0], img_res[1]])
            self.rgb = np.zeros((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4))
            self.dep = np.ones((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0]))
            self.seg = np.zeros((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0]))
            self.observation_space = self._observationSpace()

        # NOTE: EGL GPU rendering removed from env init — 8 SubprocVecEnv workers
        # each creating EGL contexts crashed the GPU driver. Use CPU TinyRenderer
        # in workers; EGL can be loaded externally for single-env eval if needed.

    ############################################################################

    def reset(self, seed=None, options=None):
        """Reset environment with new forest layout."""
        for tree_id in self.tree_ids:
            p.removeBody(tree_id, physicsClientId=self.CLIENT)
        self.tree_ids = []
        self.tree_positions = []
        self._prev_min_x = None
        self._milestones = [[False] * 4 for _ in range(self.NUM_DRONES)]

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    ############################################################################

    def _addObstacles(self):
        """Generate procedural forest with protected exit zone."""
        self.tree_positions = []
        self.tree_ids = []

        # Protected zones: no trees near drone starts or exit corridor
        protected_positions = []
        if self.INIT_XYZS is not None:
            for pos in self.INIT_XYZS:
                protected_positions.append(pos[:2])
        protected_radius = 0.45

        # Generate trees via rejection sampling
        np.random.seed(None)
        attempts = 0
        max_attempts = self.NUM_TREES * 20

        while len(self.tree_positions) < self.NUM_TREES and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(self.FOREST_X_RANGE[0], self.FOREST_X_RANGE[1])
            y = np.random.uniform(self.FOREST_Y_RANGE[0], self.FOREST_Y_RANGE[1])

            # Protected exit zone: no trees for x > 2.0
            if x > 2.0:
                continue

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
                    physicsClientId=self.CLIENT,
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=self.TREE_RADIUS,
                    length=self.TREE_HEIGHT,
                    rgbaColor=[0.35, 0.22, 0.08, 1.0],
                    physicsClientId=self.CLIENT,
                )
                tree_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_shape,
                    baseVisualShapeIndex=vis_shape,
                    basePosition=pos_3d,
                    physicsClientId=self.CLIENT,
                )
                self.tree_ids.append(tree_id)

                # Green canopy
                canopy_vis = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=self.TREE_RADIUS * 3,
                    rgbaColor=[0.1, 0.5, 0.1, 0.7],
                    physicsClientId=self.CLIENT,
                )
                canopy_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=canopy_vis,
                    basePosition=[x, y, self.TREE_HEIGHT + self.TREE_RADIUS],
                    physicsClientId=self.CLIENT,
                )
                self.tree_ids.append(canopy_id)

        # Forest-colored ground
        p.changeVisualShape(
            p.loadURDF("plane.urdf", physicsClientId=self.CLIENT),
            -1,
            rgbaColor=[0.2, 0.35, 0.1, 1.0],
            physicsClientId=self.CLIENT,
        )

    ############################################################################

    def _observationSpace(self):
        """Dict observation: per-drone RGBD vision + flattened state vector."""
        return spaces.Dict({
            "vision": spaces.Box(
                low=0, high=255,
                shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4),
                dtype=np.uint8,
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.NUM_DRONES * self.STATE_OBS_DIM,),
                dtype=np.float32,
            ),
        })

    ############################################################################

    def _computeObs(self):
        """Compute vision + state observation.

        Vision: RGBD images from each drone's forward camera (cached at 24fps).
            Channels 0-2: RGB, Channel 3: linearized depth (255=near, 0=far).
            Depth linearized from z-buffer: near objects (<2m) → 255, far (>2m) → 0.
        State layout per drone (29D), flattened to 87D:
            0:12  - own kinematics (pos, rpy, vel, ang_vel)
            12:18 - relative positions of other drones
            18    - goal x-progress (normalized)
            19:28 - nearest 3 obstacles relative pos
            28    - normalized time remaining
        """
        # Capture drone camera images (at IMG_CAPTURE_FREQ rate)
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
            for i in range(self.NUM_DRONES):
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                # Replace alpha channel with linearized depth (RGBD)
                # dep is normalized z-buffer [0,1]; linearize to metric distance
                near = self.L  # ~0.047m (drone arm length)
                far = 1000.0
                metric_depth = near * far / (far - self.dep[i] * (far - near))
                # Map 0-2m → 255-0 (near=bright, far=dark), clip >2m to 0
                depth_uint8 = np.clip((1.0 - metric_depth / 2.0) * 255, 0, 255).astype(np.uint8)
                self.rgb[i, :, :, 3] = depth_uint8

        # State vector (same as before)
        state_obs = np.zeros((self.NUM_DRONES, self.STATE_OBS_DIM), dtype=np.float32)
        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]

        x_range = self.ESCAPE_THRESHOLD - (-2.5)  # 5.0

        for i in range(self.NUM_DRONES):
            pos_i = states[i][0:3]

            # Own kinematics (12D)
            state_obs[i, 0:3] = states[i][0:3]
            state_obs[i, 3:6] = states[i][7:10]
            state_obs[i, 6:9] = states[i][10:13]
            state_obs[i, 9:12] = states[i][13:16]

            # Relative positions of other drones (6D)
            others = [j for j in range(self.NUM_DRONES) if j != i]
            for k, j in enumerate(others):
                state_obs[i, 12 + k * 3:12 + (k + 1) * 3] = states[j][0:3] - pos_i

            # Goal x-progress normalized to [0, 1] (1D)
            state_obs[i, 18] = np.clip((pos_i[0] - (-2.5)) / x_range, 0.0, 1.0)

            # Nearest 3 obstacles relative positions (9D)
            if len(self.tree_positions) > 0:
                tree_arr = np.array(self.tree_positions)
                dists = np.linalg.norm(tree_arr - pos_i, axis=1)
                nearest_idx = np.argsort(dists)[:self.NUM_NEAREST_OBSTACLES]
                for k, idx in enumerate(nearest_idx):
                    state_obs[i, 19 + k * 3:19 + (k + 1) * 3] = tree_arr[idx] - pos_i

            # Normalized time remaining (1D)
            time_elapsed = self.step_counter / (self.EPISODE_LEN_SEC * self.PYB_FREQ)
            state_obs[i, 28] = max(0.0, 1.0 - time_elapsed)

        return {
            "vision": self.rgb.astype(np.uint8),
            "state": state_obs.flatten(),
        }

    ############################################################################

    def _computeReward(self):
        """v10 reward: fix reward exploitation (backward/sacrifice-drone exploits).

        Changes from v9b:
        - min-x progress replaces centroid (slowest drone drives reward)
        - Per-drone backward penalty (-0.5) instead of clipping vel to [-0.5, 0.3]
        - Per-drone x milestone bonuses (4 checkpoints × 10.0)
        - Formation spread penalty (>1m x-spread penalized)
        - Tighter height band (0.4–1.5m vs 0.3–2.0m)

        Per-step balance (3 drones, ctrl_freq=48):
        - Hovering:              0 + 0 - 0.2(idle) - 0.1(time)           = -0.3
        - All forward 0.3 m/s:  0.63 + 0.9 - 0.1(time)                  = +1.43
        - D0 backward, D2 fwd:  min_x drops → -2.0 - 0.5 + 0.3 - 0.1   = -2.3
        - Forward near tree:    +1.43 - 1.2(prox)                        = +0.23
        - Drone hits milestone:  +10.0 one-time bonus
        """
        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        positions = np.array([s[0:3] for s in states])

        reward = 0.0

        # --- Team x-progress based on MINIMUM x (slowest drone) ---
        min_x = min(s[0] for s in states)
        if self._prev_min_x is not None:
            delta_min_x = min_x - self._prev_min_x
            reward += delta_min_x * 100.0
        self._prev_min_x = min_x

        # --- Per-drone velocity: backward penalty / forward bonus ---
        for i in range(self.NUM_DRONES):
            vel_x = states[i][10]
            if vel_x < -0.05:
                reward -= 0.5  # retreating is costly
            else:
                reward += np.clip(vel_x, 0.0, 0.3)  # only reward forward velocity

        # --- Idle penalty (replaces survival bonus — hovering is negative) ---
        avg_vel_x = np.mean([states[i][10] for i in range(self.NUM_DRONES)])
        if avg_vel_x < 0.1:
            reward -= 0.2

        # --- Per-agent penalties ---
        for i in range(self.NUM_DRONES):
            pos_i = positions[i]

            # Tree collision & proximity warning
            if len(self.tree_positions) > 0:
                tree_arr = np.array(self.tree_positions)
                dists_2d = np.linalg.norm(tree_arr[:, :2] - pos_i[:2], axis=1)
                min_dist = np.min(dists_2d)
                if min_dist < self.COLLISION_DIST:
                    reward -= 4.0
                elif min_dist < 0.45:
                    reward -= (0.45 - min_dist) * 8.0

            # Height penalty (tighter band: 0.4–1.5m)
            z = pos_i[2]
            if z < 0.4 or z > 1.5:
                reward -= 0.5

        # --- Per-drone x milestone bonuses ---
        for i in range(self.NUM_DRONES):
            for m_idx, threshold in enumerate(self.MILESTONES):
                if states[i][0] > threshold and not self._milestones[i][m_idx]:
                    reward += 10.0
                    self._milestones[i][m_idx] = True

        # --- Formation spread penalty ---
        x_positions = [s[0] for s in states]
        x_spread = max(x_positions) - min(x_positions)
        if x_spread > 1.0:
            reward -= (x_spread - 1.0) * 0.3

        # --- Time penalty ---
        reward -= 0.1

        # --- Sparse escape bonus ---
        num_escaped = sum(1 for pos in positions if pos[0] > self.ESCAPE_THRESHOLD)
        if num_escaped == self.NUM_DRONES:
            reward += 100.0

        return reward

    ############################################################################

    def _computeTerminated(self):
        """Episode terminates when ALL drones have escaped (x > threshold)."""
        for i in range(self.NUM_DRONES):
            pos = self._getDroneStateVector(i)[0:3]
            if pos[0] <= self.ESCAPE_THRESHOLD:
                return False
        return True

    ############################################################################

    def _computeTruncated(self):
        """Truncate on out-of-bounds, excessive tilt, or timeout."""
        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]

        for i in range(self.NUM_DRONES):
            pos = states[i][0:3]
            rpy = states[i][7:10]

            if abs(pos[0]) > 4.0 or abs(pos[1]) > 3.0 or pos[2] > 2.5 or pos[2] < 0.05:
                return True
            if abs(rpy[0]) > 1.2 or abs(rpy[1]) > 1.2:
                return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    ############################################################################

    def _computeInfo(self):
        """Return escape progress metrics (v10: min-x based)."""
        positions = np.array([self._getDroneStateVector(i)[0:3]
                              for i in range(self.NUM_DRONES)])
        min_x = float(np.min(positions[:, 0]))
        num_escaped = sum(1 for pos in positions if pos[0] > self.ESCAPE_THRESHOLD)

        # x-progress: based on min-x (slowest drone), normalized from start (-2.5) to escape (2.5)
        x_range = self.ESCAPE_THRESHOLD - (-2.5)
        x_progress = np.clip((min_x - (-2.5)) / x_range, 0.0, 1.0)

        min_obs_dist = float('inf')
        if len(self.tree_positions) > 0:
            tree_arr = np.array(self.tree_positions)
            for i in range(self.NUM_DRONES):
                dists_2d = np.linalg.norm(tree_arr[:, :2] - positions[i, :2], axis=1)
                min_obs_dist = min(min_obs_dist, np.min(dists_2d))

        return {
            "num_escaped": num_escaped,
            "min_x": min_x,
            "x_progress": float(x_progress),
            "all_escaped": num_escaped == self.NUM_DRONES,
            "min_obstacle_dist": min_obs_dist,
        }
