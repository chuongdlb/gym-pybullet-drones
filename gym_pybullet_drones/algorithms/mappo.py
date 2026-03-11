"""MAPPO: Multi-Agent PPO with Centralized Training, Decentralized Execution.

Decentralized actors (per-drone vision + cross-attention) with a centralized
critic (concatenated states only). N-invariant via attention pooling.

Reuses CNN, cross-attention, and gated fusion components from mavsac.py.

Architecture
------------
Actor (decentralized):
    Per-drone: SharedCNN(rgb_i) + StateMLP(state_i) → CrossAttention → GatedFusion
    All drones: AttentionPooling(fused_0..N) → features_dim

Critic (centralized):
    Concat(state_0..N) → MLP → features_dim

Usage
-----
    from gym_pybullet_drones.algorithms.mappo import MAPPOPolicy
    model = PPO(MAPPOPolicy, env, device='cuda', policy_kwargs=dict(
        features_extractor_class=ScalableVisionExtractor,
        features_extractor_kwargs=dict(embed_dim=128, num_heads=4, features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    ))
"""
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from gym_pybullet_drones.algorithms.mavsac import (
    SharedVisionCNN,
    SpatialPositionalEncoding,
    CrossAttentionBlock,
    GatedFusion,
)


class AttentionPooling(nn.Module):
    """N-invariant pooling via learned query attending to N agent features.

    Replaces the N-hardcoded Linear(embed_dim * N, features_dim) output
    projection. Works for any number of agents at inference time.
    """

    def __init__(self, embed_dim: int, features_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, agent_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        agent_features : Tensor (B, N, embed_dim)

        Returns
        -------
        Tensor (B, features_dim)
        """
        B = agent_features.shape[0]
        query = self.query.expand(B, -1, -1)  # (B, 1, embed_dim)
        pooled, _ = self.attn(query, agent_features, agent_features)  # (B, 1, embed_dim)
        return self.proj(pooled.squeeze(1))  # (B, features_dim)


class CentralizedCriticExtractor(BaseFeaturesExtractor):
    """Centralized critic: processes concatenated states from all agents.

    Ignores vision — only uses the "state" key from Dict observations.
    Uses actual N from observation space (not padded to max_agents).
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        state_space = observation_space["state"]
        num_agents, state_dim = state_space.shape
        self.num_agents = num_agents
        self.state_dim = state_dim
        input_dim = num_agents * state_dim  # 3*29=87, not 8*29=232

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        observations : dict with "state" key, Tensor (B, N, state_dim)

        Returns
        -------
        Tensor (B, features_dim)
        """
        state = observations["state"]  # (B, N, state_dim)
        B = state.shape[0]
        flat = state.reshape(B, -1)  # (B, N * state_dim)
        return self.mlp(flat)


class StateAttentionExtractor(BaseFeaturesExtractor):
    """State-only actor extractor with cross-attention between agents.

    No vision processing — uses state observations only.
    Cross-attention enables inter-agent communication via spatial positions.
    N-invariant via attention pooling.
    """

    def __init__(self, observation_space: spaces.Dict,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.embed_dim = embed_dim

        state_space = observation_space["state"]
        _, state_dim = state_space.shape

        # Per-agent state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        # Cross-attention communication
        self.cross_attention = CrossAttentionBlock(embed_dim, num_heads)

        # Gated fusion
        self.fusion = GatedFusion(embed_dim)

        # Feed-forward with residual
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        # N-invariant output pooling
        self.output_pool = AttentionPooling(embed_dim, features_dim, num_heads)

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        observations : dict
            "state": Tensor (B, N, state_dim)

        Returns
        -------
        Tensor (B, features_dim)
        """
        state = observations["state"]

        # Encode state per agent
        local_enc = self.state_encoder(state)  # (B, N, embed_dim)

        # Spatial relative positions for cross-attention
        positions = state[:, :, 0:3]
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)

        # Cross-attention
        comm_enc = self.cross_attention(local_enc, rel_pos)

        # Gated fusion
        fused = self.fusion(local_enc, comm_enc)

        # Feed-forward with residual
        ff_out = self.ff(fused)
        fused = self.ff_norm(fused + ff_out)

        # N-invariant attention pooling
        return self.output_pool(fused)


class ScalableVisionExtractor(BaseFeaturesExtractor):
    """N-invariant vision extractor: CNN + cross-attention + attention pooling.

    Same pipeline as CrossAttentionVisionExtractor but with AttentionPooling
    instead of the N-hardcoded Linear output projection.
    """

    def __init__(self, observation_space: spaces.Dict,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 features_dim: int = 256,
                 vision_out_dim: int = 64,
                 state_out_dim: int = 32):
        super().__init__(observation_space, features_dim)

        self.embed_dim = embed_dim

        vision_space = observation_space["vision"]
        state_space = observation_space["state"]
        _, img_h, img_w, img_c = vision_space.shape
        _, state_dim = state_space.shape

        # Shared vision encoder
        self.vision_cnn = SharedVisionCNN(
            img_height=img_h, img_width=img_w,
            img_channels=img_c, out_dim=vision_out_dim,
        )

        # Per-agent state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_out_dim),
            nn.ReLU(),
        )

        # Projection: vision + state → embed_dim
        self.multimodal_proj = nn.Sequential(
            nn.Linear(vision_out_dim + state_out_dim, embed_dim),
            nn.ReLU(),
        )

        # Cross-attention communication
        self.cross_attention = CrossAttentionBlock(embed_dim, num_heads)

        # Gated fusion
        self.fusion = GatedFusion(embed_dim)

        # Feed-forward with residual
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        # N-invariant output pooling
        self.output_pool = AttentionPooling(embed_dim, features_dim, num_heads)

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        observations : dict
            "vision": Tensor (B, N, H, W, C)
            "state":  Tensor (B, N, state_dim)

        Returns
        -------
        Tensor (B, features_dim)
        """
        vision = observations["vision"]
        state = observations["state"]
        B = vision.shape[0]
        N = vision.shape[1]  # Infer N dynamically

        # --- Encode vision ---
        if vision.dim() == 5 and vision.shape[-1] in (3, 4):
            vision = vision.reshape(B * N, *vision.shape[2:])
            vision = vision.permute(0, 3, 1, 2)  # NHWC → NCHW
        elif vision.dim() == 5:
            vision = vision.reshape(B * N, *vision.shape[2:])

        vision = vision.float() / 255.0
        vision_feat = self.vision_cnn(vision).view(B, N, -1)

        # --- Encode state ---
        state_feat = self.state_encoder(state)

        # --- Combine → multi-modal embedding ---
        combined = torch.cat([vision_feat, state_feat], dim=-1)
        local_enc = self.multimodal_proj(combined)

        # --- Spatial relative positions ---
        positions = state[:, :, 0:3]
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)

        # --- Cross-attention ---
        comm_enc = self.cross_attention(local_enc, rel_pos)

        # --- Gated fusion ---
        fused = self.fusion(local_enc, comm_enc)

        # --- Feed-forward with residual ---
        ff_out = self.ff(fused)
        fused = self.ff_norm(fused + ff_out)

        # --- N-invariant attention pooling ---
        return self.output_pool(fused)


class MAPPOPolicy(ActorCriticPolicy):
    """MAPPO Policy: decentralized actor (vision) + centralized critic (state).

    Uses separate feature extractors for actor and critic:
    - Actor: ScalableVisionExtractor (CNN + cross-attention + attention pooling)
    - Critic: CentralizedCriticExtractor (concatenated states → MLP)

    Both output features_dim=256 for compatibility with SB3's MlpExtractor.
    """

    def __init__(self, *args, **kwargs):
        # Ensure we use separate extractors for actor and critic
        kwargs.setdefault('share_features_extractor', False)

        # Default actor extractor
        kwargs.setdefault('features_extractor_class', StateAttentionExtractor)
        kwargs.setdefault('features_extractor_kwargs', dict(
            embed_dim=128, num_heads=4, features_dim=256,
        ))

        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        """Build policy with separate extractors for actor and critic.

        CRITICAL: We must rebuild the optimizer after replacing
        vf_features_extractor, otherwise the new module's parameters
        receive zero gradient updates.
        """
        super()._build(lr_schedule)

        # Replace the value function's feature extractor with centralized critic
        obs_space = self.observation_space
        features_dim = self.features_extractor_kwargs.get('features_dim', 256)

        self.vf_features_extractor = CentralizedCriticExtractor(
            obs_space, features_dim=features_dim,
        ).to(self.device)

        # Rebuild optimizer to include the new vf_features_extractor params
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
