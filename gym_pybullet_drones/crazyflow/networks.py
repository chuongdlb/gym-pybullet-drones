"""Flax actor-critic networks for MAPPO training.

Phase 1: MLP actor-critic (single drone).
Phase 2: Attention-based actor with centralized critic (multi-drone).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class MLP(nn.Module):
    """Simple MLP with ReLU activations."""
    hidden_dims: Sequence[int] = (256, 256)
    out_dim: int = 1

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class Actor(nn.Module):
    """Gaussian policy: obs -> (mean, log_std) for continuous actions.

    Per-drone actor with shared weights across drones.
    Input: obs_i (obs_dim,) for a single drone.
    Output: action mean (act_dim,).
    log_std is a learnable parameter (state-independent).
    """
    hidden_dims: Sequence[int] = (256, 256)
    act_dim: int = 3

    @nn.compact
    def __call__(self, obs):
        mean = MLP(self.hidden_dims, self.act_dim)(obs)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(-0.5),
            (self.act_dim,),
        )
        return mean, log_std


class Critic(nn.Module):
    """Centralized value function: concat(all_obs) -> V(s).

    For single-drone Phase 1: input = obs_i (obs_dim,).
    For multi-drone Phase 2: input = concat(obs_0, ..., obs_{n-1}).
    """
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs):
        v = MLP(self.hidden_dims, 1)(obs)
        return v.squeeze(-1)


# ============ Phase 2: Attention-based Multi-Agent ============


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention for drone token communication."""
    n_heads: int = 2
    embed_dim: int = 128

    @nn.compact
    def __call__(self, tokens):
        """Apply self-attention across drone tokens.

        Args:
            tokens: (n_drones, embed_dim) drone feature tokens
        Returns:
            attended: (n_drones, embed_dim)
        """
        head_dim = self.embed_dim // self.n_heads
        n_drones = tokens.shape[0]

        # QKV projections
        q = nn.Dense(self.embed_dim)(tokens)  # (D, E)
        k = nn.Dense(self.embed_dim)(tokens)
        v = nn.Dense(self.embed_dim)(tokens)

        # Reshape for multi-head: (D, H, head_dim)
        q = q.reshape(n_drones, self.n_heads, head_dim)
        k = k.reshape(n_drones, self.n_heads, head_dim)
        v = v.reshape(n_drones, self.n_heads, head_dim)

        # Attention scores: (H, D, D)
        scale = jnp.sqrt(head_dim).astype(q.dtype)
        attn = jnp.einsum("dhk,ehk->hde", q, k) / scale
        attn = jax.nn.softmax(attn, axis=-1)

        # Attend: (D, H, head_dim)
        out = jnp.einsum("hde,ehk->dhk", attn, v)
        out = out.reshape(n_drones, self.embed_dim)

        # Output projection + residual + LayerNorm
        out = nn.Dense(self.embed_dim)(out)
        out = nn.LayerNorm()(out + tokens)
        return out


class AttentionActor(nn.Module):
    """Per-drone actor with self-attention communication.

    1. Per-drone encoder (shared): obs_i -> 128D token
    2. Self-attention across drone tokens (2 heads)
    3. Per-drone action head: attended_token -> action mean + log_std

    Input: all_obs (n_drones, obs_dim)
    Output: means (n_drones, act_dim), log_std (act_dim,)
    """
    encoder_dim: int = 128
    act_dim: int = 3
    n_heads: int = 2

    @nn.compact
    def __call__(self, all_obs):
        """Forward pass for all drones.

        Args:
            all_obs: (n_drones, obs_dim)
        Returns:
            means: (n_drones, act_dim)
            log_std: (act_dim,)
        """
        n_drones = all_obs.shape[0]

        # Shared encoder: obs -> token
        encoder = MLP((self.encoder_dim,), self.encoder_dim)
        tokens = jax.vmap(encoder)(all_obs)  # (D, encoder_dim)

        # Self-attention
        tokens = SelfAttentionBlock(
            n_heads=self.n_heads, embed_dim=self.encoder_dim
        )(tokens)

        # Per-drone action head (shared weights)
        action_head = nn.Dense(self.act_dim)
        means = jax.vmap(action_head)(tokens)  # (D, act_dim)

        # Shared log_std
        log_std = self.param(
            "log_std",
            nn.initializers.constant(-0.5),
            (self.act_dim,),
        )

        return means, log_std


class CentralizedCritic(nn.Module):
    """Centralized critic with per-drone value outputs.

    1. Per-drone encoder (shared): obs_i -> 128D token
    2. Self-attention across tokens (each token sees all drones)
    3. Per-drone value head: attended_token_i -> Dense(256) -> Dense(1)

    After attention, each drone's token contains global information,
    enabling centralized value estimation with per-drone outputs.

    Input: all_obs (n_drones, obs_dim)
    Output: values (n_drones,) — per-drone value estimates
    """
    encoder_dim: int = 128
    hidden_dim: int = 256
    n_heads: int = 2

    @nn.compact
    def __call__(self, all_obs):
        """Forward pass.

        Args:
            all_obs: (n_drones, obs_dim)
        Returns:
            values: (n_drones,) per-drone values from global state
        """
        # Shared encoder
        encoder = MLP((self.encoder_dim,), self.encoder_dim)
        tokens = jax.vmap(encoder)(all_obs)  # (D, encoder_dim)

        # Self-attention (each token now contains global info)
        tokens = SelfAttentionBlock(
            n_heads=self.n_heads, embed_dim=self.encoder_dim
        )(tokens)

        # Per-drone value head (shared weights)
        value_head = MLP((self.hidden_dim,), 1)
        values = jax.vmap(value_head)(tokens)  # (D, 1)
        return values.squeeze(-1)  # (D,)


# ============ Unified ActorCritic ============


class ActorCritic(nn.Module):
    """Combined actor-critic for MAPPO.

    For n_drones=1: Uses simple MLP actor + critic (Phase 1).
    For n_drones>1: Uses attention actor + centralized critic (Phase 2).

    The interface is per-drone: call with single drone obs for Phase 1,
    or batch with vmap for multi-drone.
    """
    hidden_dims: Sequence[int] = (256, 256)
    act_dim: int = 3
    n_drones: int = 1

    def setup(self):
        self.actor = Actor(self.hidden_dims, self.act_dim)
        self.critic = Critic(self.hidden_dims)

    def __call__(self, obs):
        """Forward pass for a single drone's observation (Phase 1).

        Args:
            obs: (obs_dim,) single drone observation
        Returns:
            mean, log_std, value
        """
        mean, log_std = self.actor(obs)
        value = self.critic(obs)
        return mean, log_std, value

    def get_action_and_value(self, obs, rng_key):
        """Sample action and compute log_prob + value.

        Args:
            obs: (obs_dim,) single drone obs
            rng_key: JAX PRNG key
        Returns:
            action, log_prob, value, entropy
        """
        mean, log_std, value = self(obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng_key, mean.shape)
        action = mean + std * noise
        log_prob = -0.5 * (
            ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        log_prob = log_prob.sum(-1)
        entropy = 0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std).sum(-1)
        return action, log_prob, value, entropy

    def evaluate_action(self, obs, action):
        """Compute log_prob, value, entropy for a given action.

        Args:
            obs: (obs_dim,) single drone obs
            action: (act_dim,) action taken
        Returns:
            log_prob, value, entropy
        """
        mean, log_std, value = self(obs)
        std = jnp.exp(log_std)
        log_prob = -0.5 * (
            ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        log_prob = log_prob.sum(-1)
        entropy = 0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std).sum(-1)
        return log_prob, value, entropy


class MultiAgentActorCritic(nn.Module):
    """Multi-agent actor-critic with attention (Phase 2).

    Uses AttentionActor for per-drone actions with communication,
    and CentralizedCritic for shared value estimation.
    """
    encoder_dim: int = 128
    hidden_dim: int = 256
    act_dim: int = 3
    n_heads: int = 2

    def setup(self):
        self.actor = AttentionActor(
            encoder_dim=self.encoder_dim,
            act_dim=self.act_dim,
            n_heads=self.n_heads,
        )
        self.critic = CentralizedCritic(
            encoder_dim=self.encoder_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
        )

    def get_actions_and_value(self, all_obs, rng_key):
        """Get actions for all drones and per-drone values.

        Args:
            all_obs: (n_drones, obs_dim)
            rng_key: JAX PRNG key
        Returns:
            actions: (n_drones, act_dim)
            log_probs: (n_drones,)
            values: (n_drones,) per-drone values from centralized critic
            entropy: scalar (mean across drones)
        """
        means, log_std = self.actor(all_obs)  # (D, act_dim), (act_dim,)
        value = self.critic(all_obs)  # (D,)

        std = jnp.exp(log_std)
        n_drones = all_obs.shape[0]
        keys = jax.random.split(rng_key, n_drones)
        noise = jax.vmap(lambda k: jax.random.normal(k, (self.act_dim,)))(keys)

        actions = means + std[None, :] * noise  # (D, act_dim)
        log_probs = -0.5 * (
            ((actions - means) / std[None, :]) ** 2
            + 2 * log_std[None, :]
            + jnp.log(2 * jnp.pi)
        )
        log_probs = log_probs.sum(-1)  # (D,)
        entropy = 0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std).sum(-1)

        return actions, log_probs, value, entropy

    def evaluate_actions(self, all_obs, actions):
        """Evaluate given actions for all drones.

        Args:
            all_obs: (n_drones, obs_dim)
            actions: (n_drones, act_dim)
        Returns:
            log_probs: (n_drones,)
            values: (n_drones,) per-drone values
            entropy: scalar
        """
        means, log_std = self.actor(all_obs)
        value = self.critic(all_obs)  # (D,)

        std = jnp.exp(log_std)
        log_probs = -0.5 * (
            ((actions - means) / std[None, :]) ** 2
            + 2 * log_std[None, :]
            + jnp.log(2 * jnp.pi)
        )
        log_probs = log_probs.sum(-1)
        entropy = 0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std).sum(-1)

        return log_probs, value, entropy

    def get_value(self, all_obs):
        """Get per-drone values from centralized critic.

        Args:
            all_obs: (n_drones, obs_dim)
        Returns:
            values: (n_drones,) per-drone values
        """
        return self.critic(all_obs)

    def get_action_dist(self, all_obs):
        """Get actor means and log_std only (for logging).

        Args:
            all_obs: (n_drones, obs_dim)
        Returns:
            means: (n_drones, act_dim)
            log_std: (act_dim,)
        """
        return self.actor(all_obs)
