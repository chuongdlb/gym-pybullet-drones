import numpy as np
import pybullet as p
import pkg_resources

from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class GateAviary(BaseRLAviary):
    """Single agent RL problem: fly through randomly positioned gates."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 40 # Increased for 5 gates
        self.NUM_GATES = 5
        self.GATE_IDS = []
        self.gate_positions = []
        self.gate_orientations = []
        self.gates_passed = []
        self.next_gate_index = 0
        self.GATE_PASSING_THRESHOLD = 0.4  # Reduced from 0.6 due to scaling
        
        if initial_xyzs is None:
            initial_xyzs = np.array([[0, 0, 0.5]]) # Start at 0.5m height to avoid ground collision
            
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def reset(self, seed=None, options=None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Additional options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        # Remove old gates if they exist
        for gate_id in self.GATE_IDS:
            p.removeBody(gate_id, physicsClientId=self.CLIENT)
        
        self.GATE_IDS = []
        self.gate_positions = []
        self.gate_orientations = []
        self.gates_passed = [False] * self.NUM_GATES
        self.next_gate_index = 0
        
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)
        
        return obs, info

    ################################################################################
    
    def _addObstacles(self):
        """Add gates to the environment at random positions.

        Overrides BaseRLAviary's method.

        """
        # Set random seed if needed for reproducibility
        if hasattr(self, '_seed'):
            np.random.seed(self._seed)
        
        # Define spawn area boundaries
        min_x, max_x = -2, 2
        min_y, max_y = -2, 2
        min_z, max_z = 0.7, 1.5
        
        # Minimum distance between gates
        min_gate_distance = 1.5
        
        # Fix Z height for all gates 
        gate_height = 1.0 # Static height
        
        # Figure-8 Track Generation
        # Equation: x = A * sin(t), y = A * sin(t) * cos(t)
        
        scale_A = 2.0 
        
        # Static track - no random rotation
        track_rotation_angle = 0.0
        
        # Distribute 5 gates along the path
        # We avoid t=0 (start) to give drone some room
        # Range t: [0.5, 2*pi - 0.5] roughly covers the track
        t_values = np.linspace(0.5, 2 * np.pi - 0.5, self.NUM_GATES)
        
        for t in t_values:
            # Parametric position (shape of 8)
            # Using Lemniscate of Gerono: x = A sin(t), y = A sin(t) cos(t)
            raw_x = scale_A * np.sin(t)
            raw_y = scale_A * np.sin(t) * np.cos(t)
            
            # Apply global rotation to the track
            rot_x = raw_x * np.cos(track_rotation_angle) - raw_y * np.sin(track_rotation_angle)
            rot_y = raw_x * np.sin(track_rotation_angle) + raw_y * np.cos(track_rotation_angle)
            
            position = np.array([rot_x, rot_y, gate_height])
            
            # Calculate tangent vector for orientation
            # Derivatives:
            # dx/dt = A cos(t)
            # dy/dt = A (cos^2(t) - sin^2(t)) = A cos(2t)
            dx_dt = scale_A * np.cos(t)
            dy_dt = scale_A * np.cos(2*t)
            
            # Rotate tangent vector by the same global rotation
            rot_dx = dx_dt * np.cos(track_rotation_angle) - dy_dt * np.sin(track_rotation_angle)
            rot_dy = dx_dt * np.sin(track_rotation_angle) + dy_dt * np.cos(track_rotation_angle)
            
            tangent_angle = np.arctan2(rot_dy, rot_dx)
            
            # Orient gate to be perpendicular to the path (facing the drone)
            # Gate URDF is aligned along X-axis (width). We want X-axis to be perpendicular to tangent.
            # Tangent angle is the direction of flight.
            # We want gate Normal (Y-axis) to align with Tangent.
            # So we rotate X-axis by Tangent - 90 degrees.
            orientation = p.getQuaternionFromEuler([0, 0, tangent_angle - np.pi/2])
            
            # Load gate URDF
            gate_id = p.loadURDF(
                pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                position,
                orientation,
                globalScaling=0.5, # Scale down by 50%
                physicsClientId=self.CLIENT
            )
            
            self.GATE_IDS.append(gate_id)
            self.gate_positions.append(position)
            self.gate_orientations.append(orientation)

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        
        reward = 0.0
        
        # Reward for staying alive (REMOVED to prevent hovering)
        # reward += 0.1
        
        # Check if drone passed through the next gate
        if self.next_gate_index < self.NUM_GATES:
            gate_pos = self.gate_positions[self.next_gate_index]
            distance_to_gate = np.linalg.norm(drone_pos - gate_pos)
            
            # Distance Penalty: Penalize being far from the gate
            # This forces the drone to minimize distance to getting better rewards (close to 0)
            reward -= distance_to_gate * 0.01
            
            # Check if drone passed through gate
            if not self.gates_passed[self.next_gate_index]:
                # Gate is at position gate_pos with size approximately 1.0x1.0
                # Check if drone is within gate bounds
                gate_orientation = self.gate_orientations[self.next_gate_index]
                
                # Transform drone position to gate's local frame
                euler = p.getEulerFromQuaternion(gate_orientation)
                yaw = euler[2]
                
                # Rotate drone position relative to gate
                rel_pos = drone_pos - gate_pos
                cos_yaw = np.cos(-yaw)
                sin_yaw = np.sin(-yaw)
                local_x = rel_pos[0] * cos_yaw - rel_pos[1] * sin_yaw
                local_y = rel_pos[0] * sin_yaw + rel_pos[1] * cos_yaw
                local_z = rel_pos[2]
                
                # Gate dimensions (Approximate based on URDF and 0.5 scaling)
                # Original: ~1.0m width/height. Scaled: ~0.5m.
                # Valid region: +/- 0.25 ish
                if (abs(local_x) < self.GATE_PASSING_THRESHOLD and 
                    abs(local_y) < 0.25 and 
                    abs(local_z) < 0.25):
                    # Passed through gate!
                    self.gates_passed[self.next_gate_index] = True
                    self.next_gate_index += 1
                    reward += 100.0  # Big reward for passing gate
                    
                    # Extra reward for passing all gates
                    if self.next_gate_index >= self.NUM_GATES:
                        reward += 200.0
            
            # Progress Reward: Speed towards the goal
            # Project velocity vector onto the direction to the gate
            vel = state[10:13]
            if np.linalg.norm(vel) > 0.1:
                direction_to_gate = gate_pos - drone_pos
                dist = np.linalg.norm(direction_to_gate)
                if dist > 0:
                    dir_normalized = direction_to_gate / dist
                    vel_towards_gate = np.dot(vel, dir_normalized)
                    # Reward for speed towards gate, but cap it to avoid exploitation
                    reward += vel_towards_gate * 0.1
            
            # Time penalty to encourage speed (Reduced to prevent panic)
            reward -= 0.01 
            
            # Facing Reward: Encourage pointing usage towards the goal
            # This helps the drone filter out actions that turn it away
            # Quaternion is at state[3:7]
            rotation = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
            heading_vector = rotation[:, 0] # Body X-axis
            
            direction_to_gate = gate_pos - drone_pos
            dist = np.linalg.norm(direction_to_gate)
            if dist > 0:
                dir_normalized = direction_to_gate / dist
                # Reward for facing the gate
                heading_dot = np.dot(heading_vector, dir_normalized)
                reward += heading_dot * 0.05 

            # Penalty for flying too high above the target gate
            # The gate center is at gate_pos. The gate height is roughly 1m (frame size).
            # If drone is significantly higher than gate center + 0.5, apply penalty.
            z_diff = drone_pos[2] - gate_pos[2]
            if z_diff > 0.5: 
                # drone is above the top rim of the gate (assuming gate is ~1m tall centered at Z)
                reward -= z_diff * 1.0 # Tune this coefficient as needed
        
        # Penalty for tilting too much (Relaxed)
        # REMOVED intermediate penalty to prevent early termination "suicide"
        # if abs(state[7]) > 1.0 or abs(state[8]) > 1.0:
        #    reward -= 1.0
        
        # Penalty for going out of bounds
        if abs(drone_pos[0]) > 3 or abs(drone_pos[1]) > 3 or drone_pos[2] < 0.1 or drone_pos[2] > 2.5:
            reward -= 10.0 # Increased from -2.0 to prevent "suicide" strategies
        
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # Episode terminates successfully if all gates are passed
        if self.next_gate_index >= self.NUM_GATES:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        
        # Truncate when the drone is too far away
        if (abs(state[0]) > 3.5 or abs(state[1]) > 3.5 or 
            state[2] > 3.0 or state[2] < 0.05):
            # print(f"[DEBUG] Truncated: Bounds/Ground (Pos: {state[0:3]})")
            return True
        
        # Truncate when the drone is too tilted (Relaxed for racing)
        if abs(state[7]) > 1.5 or abs(state[8]) > 1.5:
            # print(f"[DEBUG] Truncated: Tilt (Roll: {state[7]:.2f}, Pitch: {state[8]:.2f})")
            return True
        
        # Truncate on timeout
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            # print("[DEBUG] Truncated: Timeout") # Less important
            return True
        
        return False

    ################################################################################
    
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        # Get the base observation space
        base_space = super()._observationSpace()
        
        # If not kinematic, return base space (only KIN is modified for now)
        if self.OBS_TYPE != ObservationType.KIN:
            return base_space
            
        # Add 3 elements for the relative position to the next gate
        lo = -np.inf
        hi = np.inf
        
        # Extend the bounds
        # Base space low/high are (NUM_DRONES, N)
        extra_low = np.array([[lo, lo, lo] for _ in range(self.NUM_DRONES)])
        extra_high = np.array([[hi, hi, hi] for _ in range(self.NUM_DRONES)])
        
        new_low = np.hstack([base_space.low, extra_low])
        new_high = np.hstack([base_space.high, extra_high])
        
        return spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        # Get the base observation
        obs = super()._computeObs()
        
        # If not kinematic, return base obs
        if self.OBS_TYPE != ObservationType.KIN:
            return obs
            
        # Calculate relative position to the next gate
        gate_rel_pos = np.zeros((self.NUM_DRONES, 3))
        
        if self.next_gate_index < self.NUM_GATES:
            target_gate_pos = self.gate_positions[self.next_gate_index]
            
            # Get drone position
            state = self._getDroneStateVector(0)
            drone_pos = state[0:3]
            
            # Relative vector (Target - Current)
            gate_rel_pos[0, :] = target_gate_pos - drone_pos
            
        # Append to observation
        # obs is (NUM_DRONES, N)
        new_obs = np.hstack([obs, gate_rel_pos])
        
        return new_obs.astype('float32')

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Returns
        -------
        dict[str, int]
            Info dictionary with gates passed.

        """
        return {
            "gates_passed": sum(self.gates_passed),
            "next_gate": self.next_gate_index,
            "all_gates_passed": self.next_gate_index >= self.NUM_GATES
        }
