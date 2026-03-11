"""Hyperparameter configuration for Crazyflow ForestEscape training."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ForestConfig:
    """Forest environment parameters."""
    n_trees: int = 35          # Max trees (array size, always allocated)
    n_trees_start: int = 5     # Curriculum: start with this many active trees
    n_trees_final: int = 35    # Curriculum: ramp up to this many
    curriculum_frac: float = 0.5  # Fraction of training to reach n_trees_final
    terminate_on_collision: bool = True  # Episode ends on first collision
    forest_x_range: tuple = (-3.0, 2.0)  # No trees for x > 2.0
    forest_y_range: tuple = (-2.0, 2.0)
    tree_radius: float = 0.06
    collision_dist: float = 0.15
    proximity_dist: float = 0.45
    proximity_coef: float = 8.0
    protected_radius: float = 0.60  # No trees near drone starts
    escape_threshold: float = 2.5
    height_min: float = 0.4
    height_max: float = 1.5


@dataclass(frozen=True)
class ObsConfig:
    """Observation parameters."""
    n_lidar_rays: int = 24
    lidar_fov_deg: float = 180.0
    lidar_max_range: float = 3.0
    k_nearest_trees: int = 8
    obs_dim_per_drone: int = 60  # 12 kin + 6 rel + 2 goal/time + 24 lidar + 16 nearest


@dataclass(frozen=True)
class RewardConfig:
    """Reward function parameters."""
    progress_coef: float = 100.0
    forward_vel_max: float = 0.5
    forward_vel_bonus: float = 3.0
    backward_threshold: float = -0.02
    backward_penalty: float = 2.0
    idle_threshold: float = 0.05
    idle_penalty: float = 0.3
    collision_penalty: float = 2.0
    height_penalty: float = 0.5
    time_penalty: float = 0.1
    milestone_bonus: float = 10.0
    milestones: tuple = (-1.5, -0.5, 0.5, 1.5)
    escape_bonus: float = 100.0
    formation_spread_threshold: float = 1.0
    formation_spread_coef: float = 0.3
    speed_proximity_coef: float = 0.0


@dataclass
class TrainConfig:
    """Training hyperparameters (MAPPO)."""
    seed: int = 42
    n_worlds: int = 1024
    n_drones: int = 1  # Phase 1: single drone
    total_timesteps: int = 5_000_000
    n_steps: int = 2048  # Steps per rollout
    n_epochs: int = 3
    minibatch_size: int = 128
    lr_init: float = 3e-4
    lr_final: float = 5e-5
    clip_eps: float = 0.2
    ent_coef: float = 0.005
    ent_coef_final: float = 0.0005  # Entropy decay target
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    # Network (Phase 1: MLP)
    hidden_dims: tuple = (256, 256)
    actor_hidden: int = 128
    critic_hidden: int = 128
    # Network (Phase 2: Attention)
    encoder_dim: int = 128
    n_heads: int = 2
    critic_hidden_dim: int = 256
    # Environment
    ctrl_freq: int = 50
    physics_freq: int = 500
    episode_len_sec: float = 30.0
    action_scale: float = 0.15
    # Logging
    eval_freq: int = 10_000
    log_freq: int = 5_000
    save_freq: int = 100_000
    save_dir: str = "results/crazyflow"

    @property
    def episode_len_steps(self) -> int:
        return int(self.episode_len_sec * self.ctrl_freq)

    @property
    def n_substeps(self) -> int:
        return self.physics_freq // self.ctrl_freq
