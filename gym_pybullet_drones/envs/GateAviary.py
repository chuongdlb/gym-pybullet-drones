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
        self.EPISODE_LEN_SEC = 20
        self.NUM_GATES = 3
        self.GATE_IDS = []
        self.gate_positions = []
        self.gate_orientations = []
        self.gates_passed = []
        self.next_gate_index = 0
        self.GATE_PASSING_THRESHOLD = 0.6  # Distance threshold to consider gate passed
        
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
        
        for i in range(self.NUM_GATES):
            # Keep trying until we find a valid position
            max_attempts = 100
            for attempt in range(max_attempts):
                # Random position
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                z = np.random.uniform(min_z, max_z)
                position = np.array([x, y, z])
                
                # Check distance from all previous gates
                valid_position = True
                for prev_pos in self.gate_positions:
                    if np.linalg.norm(position - prev_pos) < min_gate_distance:
                        valid_position = False
                        break
                
                # Check distance from origin (drone start position)
                if np.linalg.norm(position - np.array([0, 0, 0])) < 1.0:
                    valid_position = False
                
                if valid_position:
                    break
            
            # Random orientation (yaw rotation)
            yaw = np.random.uniform(0, 2 * np.pi)
            orientation = p.getQuaternionFromEuler([0, 0, yaw])
            
            # Load gate URDF
            gate_id = p.loadURDF(
                pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                position,
                orientation,
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
        
        # Reward for staying alive
        reward += 0.1
        
        # Check if drone passed through the next gate
        if self.next_gate_index < self.NUM_GATES:
            gate_pos = self.gate_positions[self.next_gate_index]
            distance_to_gate = np.linalg.norm(drone_pos - gate_pos)
            
            # Reward for moving closer to the next gate
            reward += max(0, (3.0 - distance_to_gate) * 0.5)
            
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
                
                # Gate dimensions: width ~1.0m, height ~1.0m, thickness ~0.1m
                if (abs(local_x) < self.GATE_PASSING_THRESHOLD and 
                    abs(local_y) < 0.55 and 
                    abs(local_z) < 0.45):
                    # Passed through gate!
                    self.gates_passed[self.next_gate_index] = True
                    self.next_gate_index += 1
                    reward += 100.0  # Big reward for passing gate
                    
                    # Extra reward for passing all gates
                    if self.next_gate_index >= self.NUM_GATES:
                        reward += 200.0
        
        # Penalty for tilting too much
        if abs(state[7]) > 0.5 or abs(state[8]) > 0.5:
            reward -= 1.0
        
        # Penalty for going out of bounds
        if abs(drone_pos[0]) > 3 or abs(drone_pos[1]) > 3 or drone_pos[2] < 0.1 or drone_pos[2] > 2.5:
            reward -= 2.0
        
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
            return True
        
        # Truncate when the drone is too tilted
        if abs(state[7]) > 0.8 or abs(state[8]) > 0.8:
            return True
        
        # Truncate on timeout
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
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
