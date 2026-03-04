"""MAVSAC: Multi-Agent Vision-Sharing Actor-Critic.

A novel multi-agent RL architecture for cooperative drone navigation.
Each drone captures onboard RGB images. The MAVSAC policy encodes these
images via a shared CNN, then uses cross-attention to fuse visual
observations across agents — enabling collective perception in
environments with limited individual visibility (e.g., dense forests).

Architecture
------------
For each agent i:
    v_i = SharedCNN(rgb_image_i)         # Visual encoding
    s_i = StateMLP(kinematic_state_i)    # State encoding
    h_i^local = Concat(v_i, s_i)        # Local multi-modal feature

    For each agent j:
        h_j^local = Concat(v_j, s_j)

    h_i^comm = CrossAttention(Q=h_i, K=H_all, V=H_all, bias=SpatialEncoding)
    gate = sigmoid(W_g * [h_i^local || h_i^comm])
    h_i^fused = gate * h_i^local + (1 - gate) * h_i^comm

    features = OutputProjection(Concat(h_0^fused, ..., h_N^fused))

Key Properties
--------------
- Cross-attention is permutation-equivariant: output is invariant to
  agent ordering (each agent's result depends on the set, not the order)
- Gated fusion preserves gradient flow to both local and communication streams
- Shared CNN + state encoder ensures parameter efficiency
- Spatial attention bias: relative drone positions modulate attention,
  so nearby drones communicate more strongly
- Compatible with CTDE: communication features can replace centralization

Usage with SB3
--------------
    from gym_pybullet_drones.algorithms.mavsac import MAVSACPolicy

    model = PPO(MAVSACPolicy, env, policy_kwargs=dict(
        features_extractor_kwargs=dict(
            num_agents=3, embed_dim=64, num_heads=4, features_dim=128,
        ),
    ))

"""
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class SharedVisionCNN(nn.Module):
    """Shared CNN encoder for per-drone RGB images.

    Encodes a (H, W, 4) RGBA image into a compact feature vector.
    Shared across all agents for parameter efficiency.
    """

    def __init__(self, img_height: int = 48, img_width: int = 64,
                 img_channels: int = 4, out_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, img_height, img_width)
            cnn_out_size = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, out_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : Tensor (batch * num_agents, C, H, W)

        Returns
        -------
        Tensor (batch * num_agents, out_dim)
        """
        return self.fc(self.cnn(images))


class SpatialPositionalEncoding(nn.Module):
    """Encode relative spatial positions between agents as attention bias."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, num_heads * 2),
            nn.ReLU(),
            nn.Linear(num_heads * 2, num_heads),
        )

    def forward(self, relative_positions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        relative_positions : Tensor (batch, N, N, 3)

        Returns
        -------
        Tensor (batch, num_heads, N, N)
        """
        bias = self.mlp(relative_positions)  # (B, N, N, num_heads)
        return bias.permute(0, 3, 1, 2)


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention with spatial position bias.

    Core communication mechanism: each agent attends to all agents
    with attention weights modulated by spatial proximity.
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.spatial_encoding = SpatialPositionalEncoding(num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, agent_encodings: torch.Tensor,
                relative_positions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        agent_encodings : Tensor (B, N, embed_dim)
        relative_positions : Tensor (B, N, N, 3)

        Returns
        -------
        Tensor (B, N, embed_dim)
        """
        B, N, D = agent_encodings.shape

        Q = self.q_proj(agent_encodings).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(agent_encodings).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(agent_encodings).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        spatial_bias = self.spatial_encoding(relative_positions)
        attn_scores = attn_scores + spatial_bias

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        attn_output = self.out_proj(attn_output)

        return self.layer_norm(agent_encodings + attn_output)


class GatedFusion(nn.Module):
    """Learned sigmoid gate blending local perception with communicated info."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, local_feat: torch.Tensor, comm_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([local_feat, comm_feat], dim=-1)
        gate = self.gate_net(combined)
        return gate * local_feat + (1 - gate) * comm_feat


class CrossAttentionVisionExtractor(BaseFeaturesExtractor):
    """MAVSAC Feature Extractor: CNN + Cross-Attention for shared vision.

    Processes Dict observations with "vision" (RGB images) and "state"
    (kinematic) components. Each drone's image is encoded via a shared CNN,
    combined with its state encoding, then fused across agents via
    cross-attention with spatial position bias.

    Parameters
    ----------
    observation_space : spaces.Dict
        Must contain "vision" and "state" keys.
    num_agents : int
        Number of cooperative agents (drones).
    embed_dim : int
        Embedding dimension for agent feature vectors.
    num_heads : int
        Number of attention heads for cross-attention.
    features_dim : int
        Output feature dimension for actor-critic heads.
    vision_out_dim : int
        CNN output dimension per agent.
    state_out_dim : int
        State encoder output dimension per agent.
    """

    def __init__(self, observation_space: spaces.Dict,
                 num_agents: int = 3,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 features_dim: int = 128,
                 vision_out_dim: int = 48,
                 state_out_dim: int = 16):
        super().__init__(observation_space, features_dim)

        self.num_agents = num_agents
        self.embed_dim = embed_dim

        # Extract shapes from observation space
        vision_space = observation_space["vision"]
        state_space = observation_space["state"]
        _, img_h, img_w, img_c = vision_space.shape
        _, state_dim = state_space.shape

        # Shared vision encoder (CNN)
        self.vision_cnn = SharedVisionCNN(
            img_height=img_h, img_width=img_w,
            img_channels=img_c, out_dim=vision_out_dim
        )

        # Per-agent state encoder (MLP)
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

        # Output: all agent features → combined feature vector
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * num_agents, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """Forward pass: encode vision + state → communicate → fuse → project.

        Parameters
        ----------
        observations : dict
            "vision": Tensor (B, N, H, W, C) or (B, N, C, H, W)
            "state":  Tensor (B, N, state_dim)

        Returns
        -------
        Tensor (B, features_dim)
        """
        vision = observations["vision"]  # (B, N, H, W, C)
        state = observations["state"]    # (B, N, state_dim)
        B = vision.shape[0]
        N = self.num_agents

        # --- Encode vision ---
        # Reshape: (B, N, H, W, C) → (B*N, C, H, W)
        if vision.dim() == 5 and vision.shape[-1] in (3, 4):
            # (B, N, H, W, C) → (B*N, C, H, W)
            vision = vision.reshape(B * N, *vision.shape[2:])
            vision = vision.permute(0, 3, 1, 2)  # NHWC → NCHW
        elif vision.dim() == 5:
            # Already (B, N, C, H, W)
            vision = vision.reshape(B * N, *vision.shape[2:])

        vision = vision.float() / 255.0  # normalize pixel values
        vision_feat = self.vision_cnn(vision)  # (B*N, vision_out_dim)
        vision_feat = vision_feat.view(B, N, -1)  # (B, N, vision_out_dim)

        # --- Encode state ---
        state_feat = self.state_encoder(state)  # (B, N, state_out_dim)

        # --- Combine vision + state → multi-modal embedding ---
        combined = torch.cat([vision_feat, state_feat], dim=-1)  # (B, N, vis+state)
        local_enc = self.multimodal_proj(combined)  # (B, N, embed_dim)

        # --- Spatial relative positions for attention bias ---
        # Extract XYZ positions from state (first 3 elements per agent)
        positions = state[:, :, 0:3]  # (B, N, 3)
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)

        # --- Cross-attention communication (shared vision fusion) ---
        comm_enc = self.cross_attention(local_enc, rel_pos)  # (B, N, embed_dim)

        # --- Gated fusion ---
        fused = self.fusion(local_enc, comm_enc)  # (B, N, embed_dim)

        # Feed-forward with residual
        ff_out = self.ff(fused)
        fused = self.ff_norm(fused + ff_out)

        # --- Project to output ---
        fused_flat = fused.reshape(B, -1)  # (B, N * embed_dim)
        return self.output_proj(fused_flat)  # (B, features_dim)


class MAVSACPolicy(ActorCriticPolicy):
    """MAVSAC Actor-Critic Policy for SB3 PPO.

    Uses CrossAttentionVisionExtractor to process multi-agent Dict
    observations (RGB images + kinematic state) with cross-attention
    communication between agents.

    Usage
    -----
        model = PPO(MAVSACPolicy, env, policy_kwargs=dict(
            features_extractor_class=CrossAttentionVisionExtractor,
            features_extractor_kwargs=dict(
                num_agents=3, embed_dim=64, num_heads=4, features_dim=128,
            ),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        ))
    """

    def __init__(self, *args, **kwargs):
        if 'features_extractor_class' not in kwargs:
            kwargs['features_extractor_class'] = CrossAttentionVisionExtractor
        if 'features_extractor_kwargs' not in kwargs:
            kwargs['features_extractor_kwargs'] = dict(
                num_agents=3,
                embed_dim=64,
                num_heads=4,
                features_dim=128,
            )
        super().__init__(*args, **kwargs)
