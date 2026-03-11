"""MAPPO training loop for ForestEscape on Crazyflow.

Single-file training script. Collects rollouts, computes GAE,
runs PPO updates with clipped surrogate objective.

Phase 1 (n_drones=1): MLP ActorCritic, per-drone vmap.
Phase 2 (n_drones>1): AttentionActor + CentralizedCritic, vmap over worlds.
"""

import time
import os
import json
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import flax.core
from flax.training.train_state import TrainState
from functools import partial

from .config import TrainConfig, ForestConfig, ObsConfig, RewardConfig
from .networks import ActorCritic, MultiAgentActorCritic
from .forest_escape_env import ForestEscapeEnv


# ============ Phase 1: Single-Drone (MLP) ============


def _create_single_train_state(rng_key, tc: TrainConfig, oc: ObsConfig):
    """Initialize single-drone ActorCritic and optimizer."""
    model = ActorCritic(
        hidden_dims=tc.hidden_dims,
        act_dim=3,
        n_drones=tc.n_drones,
    )
    dummy_obs = jnp.zeros(oc.obs_dim_per_drone)
    params = model.init(rng_key, dummy_obs)

    # Forward bias on actor output
    flat_params = flax.core.unfreeze(params)
    flat_params["params"]["actor"]["MLP_0"]["Dense_2"]["bias"] = jnp.array([0.5, 0.0, 0.0])
    params = flax.core.freeze(flat_params)

    return params, model


def _make_single_inference_fn(hidden_dims, act_dim, n_drones):
    """Create JIT-compiled inference for single-drone (vmap over all drones)."""
    model = ActorCritic(hidden_dims=hidden_dims, act_dim=act_dim, n_drones=n_drones)

    @jax.jit
    def get_actions_and_values(params, obs_flat, rng_keys):
        actions, log_probs, values, entropy = jax.vmap(
            lambda o, k: model.apply(params, o, k, method=model.get_action_and_value)
        )(obs_flat, rng_keys)
        return actions, log_probs, values

    @jax.jit
    def get_values(params, obs_flat):
        return jax.vmap(lambda o: model.apply(params, o)[2])(obs_flat)

    return get_actions_and_values, get_values


def _make_single_loss_fn(hidden_dims, act_dim, n_drones):
    """PPO loss for single-drone (samples are individual drone obs)."""
    model = ActorCritic(hidden_dims=hidden_dims, act_dim=act_dim, n_drones=n_drones)

    def ppo_loss(params, batch, clip_eps, ent_coef, vf_coef):
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["values"]

        new_log_probs, new_values, entropy = jax.vmap(
            lambda o, a: model.apply(params, o, a, method=model.evaluate_action)
        )(obs, actions)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = jnp.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -jnp.minimum(surr1, surr2).mean()

        v_clipped = old_values + jnp.clip(new_values - old_values, -clip_eps, clip_eps)
        v_loss1 = (new_values - returns) ** 2
        v_loss2 = (v_clipped - returns) ** 2
        value_loss = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()

        entropy_loss = -entropy.mean()
        total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        info = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
            "clip_fraction": (jnp.abs(ratio - 1.0) > clip_eps).mean(),
            "approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
        }
        return total_loss, info

    return ppo_loss


# ============ Phase 2: Multi-Drone (Attention) ============


def _create_multi_train_state(rng_key, tc: TrainConfig, oc: ObsConfig):
    """Initialize multi-drone MultiAgentActorCritic and optimizer."""
    model = MultiAgentActorCritic(
        encoder_dim=tc.encoder_dim,
        hidden_dim=tc.critic_hidden_dim,
        act_dim=3,
        n_heads=tc.n_heads,
    )
    dummy_obs = jnp.zeros((tc.n_drones, oc.obs_dim_per_drone))
    dummy_key = jax.random.PRNGKey(0)
    params = model.init(dummy_key, dummy_obs, dummy_key, method=model.get_actions_and_value)

    # Forward bias on attention actor output head
    flat_params = flax.core.unfreeze(params)
    flat_params["params"]["actor"]["Dense_0"]["bias"] = jnp.array([0.5, 0.0, 0.0])
    params = flax.core.freeze(flat_params)

    return params, model


def _make_multi_inference_fn(encoder_dim, hidden_dim, act_dim, n_heads):
    """Create JIT-compiled inference for multi-drone (vmap over worlds)."""
    model = MultiAgentActorCritic(
        encoder_dim=encoder_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        n_heads=n_heads,
    )

    @jax.jit
    def get_actions_and_values(params, obs, rng_keys):
        """Multi-drone inference.

        Args:
            params: network parameters
            obs: (n_worlds, n_drones, obs_dim)
            rng_keys: (n_worlds, 2)

        Returns:
            actions: (n_worlds, n_drones, act_dim)
            log_probs: (n_worlds, n_drones)
            values: (n_worlds, n_drones) — per-drone values from centralized critic
        """
        actions, log_probs, values, entropy = jax.vmap(
            lambda o, k: model.apply(params, o, k, method=model.get_actions_and_value)
        )(obs, rng_keys)
        return actions, log_probs, values

    @jax.jit
    def get_values(params, obs):
        """Get per-drone values from centralized critic.

        Args:
            obs: (n_worlds, n_drones, obs_dim)

        Returns:
            values: (n_worlds, n_drones) — per-drone values
        """
        return jax.vmap(
            lambda o: model.apply(params, o, method=model.get_value)
        )(obs)

    return get_actions_and_values, get_values


def _make_multi_loss_fn(encoder_dim, hidden_dim, act_dim, n_heads):
    """PPO loss for multi-drone (samples are per-world with all drones)."""
    model = MultiAgentActorCritic(
        encoder_dim=encoder_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        n_heads=n_heads,
    )

    def ppo_loss(params, batch, clip_eps, ent_coef, vf_coef):
        """Compute PPO loss for multi-agent.

        With per-drone values, we flatten all (world, drone) into individual
        samples — same as single-drone but evaluate_actions handles attention.

        Batch dims:
            obs: (B, n_drones, obs_dim) — B is (world,timestep) pairs
            actions: (B, n_drones, act_dim)
            log_probs: (B, n_drones)
            advantages: (B, n_drones)
            returns: (B, n_drones)
            values: (B, n_drones)
        """
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["values"]

        # Forward pass (vmap over batch of worlds)
        new_log_probs, new_values, entropy = jax.vmap(
            lambda o, a: model.apply(params, o, a, method=model.evaluate_actions)
        )(obs, actions)
        # new_log_probs: (B, n_drones)
        # new_values: (B, n_drones) per-drone
        # entropy: (B,) mean across drones

        # Normalize advantages across full batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (per-drone ratios, then mean over all)
        ratio = jnp.exp(new_log_probs - old_log_probs)  # (B, D)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -jnp.minimum(surr1, surr2).mean()

        # Value loss (per-drone values)
        v_clipped = old_values + jnp.clip(new_values - old_values, -clip_eps, clip_eps)
        v_loss1 = (new_values - returns) ** 2
        v_loss2 = (v_clipped - returns) ** 2
        value_loss = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()

        entropy_loss = -entropy.mean()
        total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        info = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
            "clip_fraction": (jnp.abs(ratio - 1.0) > clip_eps).mean(),
            "approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
        }
        return total_loss, info

    return ppo_loss


# ============ Shared: GAE ============


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Compute GAE advantages (reverse scan).

    Args:
        rewards: (T, W, D) per-drone rewards
        values: (T, W, D) per-drone values
        dones: (T, W) bool
        last_value: (W, D) bootstrap value
        gamma, gae_lambda: floats

    Returns:
        advantages: (T, W, D)
        returns: (T, W, D)
    """
    T = rewards.shape[0]
    n_drones = rewards.shape[2]

    def scan_fn(carry, t):
        last_gae, next_value = carry
        idx = T - 1 - t
        done = dones[idx]
        non_terminal = (1.0 - done.astype(jnp.float32))[:, None]  # (W, 1)
        delta = rewards[idx] + gamma * next_value * non_terminal - values[idx]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        return (last_gae, values[idx]), last_gae

    init_carry = (jnp.zeros((rewards.shape[1], n_drones)), last_value)
    _, advantages_rev = jax.lax.scan(scan_fn, init_carry, jnp.arange(T))

    advantages = advantages_rev[::-1]  # (T, W, D)
    returns = advantages + values  # (T, W, D)

    return advantages, returns


# ============ Shared: Update Step ============


def make_update_fn(loss_fn, clip_eps, vf_coef):
    """Create JIT-compiled update function from any loss_fn.

    ent_coef is passed dynamically to support entropy decay.
    """
    @jax.jit
    def update_step(train_state, batch, ent_coef):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, info), grads = grad_fn(
            train_state.params, batch, clip_eps, ent_coef, vf_coef,
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, info

    return update_step


# ============ Main Training Loop ============


def create_train_state(rng_key, tc: TrainConfig, oc: ObsConfig):
    """Initialize network and optimizer, branching on n_drones."""
    if tc.n_drones > 1:
        params, model = _create_multi_train_state(rng_key, tc, oc)
    else:
        params, model = _create_single_train_state(rng_key, tc, oc)

    # Linear LR schedule
    n_rollouts = tc.total_timesteps // (tc.n_steps * tc.n_worlds * tc.n_drones)
    if tc.n_drones > 1:
        samples_per_rollout = tc.n_steps * tc.n_worlds  # batch by (world, timestep)
    else:
        samples_per_rollout = tc.n_steps * tc.n_worlds * tc.n_drones
    total_updates = n_rollouts * tc.n_epochs * (samples_per_rollout // tc.minibatch_size)
    schedule = optax.linear_schedule(tc.lr_init, tc.lr_final, max(total_updates, 1))

    optimizer = optax.chain(
        optax.clip_by_global_norm(tc.max_grad_norm),
        optax.adam(learning_rate=schedule),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


def train(tc: TrainConfig = None, fc: ForestConfig = None,
          oc: ObsConfig = None, rc: RewardConfig = None):
    """Main MAPPO training loop (single-drone and multi-drone)."""
    tc = tc or TrainConfig()
    fc = fc or ForestConfig()
    oc = oc or ObsConfig()
    rc = rc or RewardConfig()

    multi_drone = tc.n_drones > 1
    phase = "Phase 2 (Attention)" if multi_drone else "Phase 1 (MLP)"

    print(f"=== ForestEscape MAPPO Training — {phase} ===")
    print(f"n_worlds={tc.n_worlds}, n_drones={tc.n_drones}")
    print(f"total_timesteps={tc.total_timesteps}")
    print(f"n_steps={tc.n_steps}, n_epochs={tc.n_epochs}")
    print(f"lr: {tc.lr_init} -> {tc.lr_final}")
    if multi_drone:
        print(f"encoder_dim={tc.encoder_dim}, n_heads={tc.n_heads}, critic_hidden={tc.critic_hidden_dim}")
    print(f"Device: {jax.devices()}")

    # Initialize
    rng_key = jax.random.PRNGKey(tc.seed)
    rng_key, init_key = jax.random.split(rng_key)

    # Create env
    env = ForestEscapeEnv(tc, fc, oc, rc)

    # Create network + optimizer
    train_state = create_train_state(init_key, tc, oc)
    param_count = sum(x.size for x in jax.tree.leaves(train_state.params))
    print(f"Parameters: {param_count:,}")

    # Create JIT-compiled functions
    if multi_drone:
        inference_fn, value_fn = _make_multi_inference_fn(
            tc.encoder_dim, tc.critic_hidden_dim, 3, tc.n_heads,
        )
        loss_fn = _make_multi_loss_fn(
            tc.encoder_dim, tc.critic_hidden_dim, 3, tc.n_heads,
        )
    else:
        inference_fn, value_fn = _make_single_inference_fn(tc.hidden_dims, 3, tc.n_drones)
        loss_fn = _make_single_loss_fn(tc.hidden_dims, 3, tc.n_drones)

    update_fn = make_update_fn(loss_fn, tc.clip_eps, tc.vf_coef)

    # Setup logging
    os.makedirs(tc.save_dir, exist_ok=True)
    log_file = os.path.join(tc.save_dir, "training_log.jsonl")

    # Reset env
    rng_key, reset_key = jax.random.split(rng_key)
    obs, forest_state = env.reset(reset_key)

    global_step = 0
    start_time = time.time()
    n_updates = 0
    steps_per_rollout = tc.n_steps * tc.n_worlds * tc.n_drones

    print(f"Curriculum: {fc.n_trees_start} -> {fc.n_trees_final} trees over {fc.curriculum_frac*100:.0f}% of training")
    print(f"Entropy: {tc.ent_coef} -> {tc.ent_coef_final}")
    print(f"Collision termination: {fc.terminate_on_collision}")

    while global_step < tc.total_timesteps:
        # Update curriculum and entropy
        progress_frac = global_step / tc.total_timesteps
        env.set_curriculum(progress_frac)

        # Linear entropy decay
        ent_frac = min(progress_frac / max(fc.curriculum_frac, 1e-6), 1.0)
        current_ent_coef = tc.ent_coef + ent_frac * (tc.ent_coef_final - tc.ent_coef)
        # ===== Collect rollout =====
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []

        rollout_min_x = []
        rollout_vel_x = []
        rollout_collisions = []
        rollout_escapes = []

        for step in range(tc.n_steps):
            rng_key, act_key = jax.random.split(rng_key)

            if multi_drone:
                # obs: (W, D, obs_dim) — pass directly, vmap over worlds
                act_keys = jax.random.split(act_key, tc.n_worlds)
                actions, log_probs, values = inference_fn(
                    train_state.params, obs, act_keys
                )
                # actions: (W, D, 3), log_probs: (W, D), values: (W, D)
                actions_clipped = jnp.clip(actions, -1.0, 1.0)
            else:
                # Flatten (W, D) -> (W*D) for batched inference
                obs_flat = obs.reshape(-1, oc.obs_dim_per_drone)
                act_keys = jax.random.split(act_key, obs_flat.shape[0])

                actions_flat, log_probs_flat, values_flat = inference_fn(
                    train_state.params, obs_flat, act_keys
                )
                actions = actions_flat.reshape(tc.n_worlds, tc.n_drones, 3)
                actions_clipped = jnp.clip(actions, -1.0, 1.0)
                log_probs = log_probs_flat.reshape(tc.n_worlds, tc.n_drones)
                values = values_flat.reshape(tc.n_worlds, tc.n_drones)

            rollout_obs.append(obs)
            rollout_actions.append(actions)  # raw for log_prob match
            rollout_log_probs.append(log_probs)
            rollout_values.append(values)

            # Step environment
            obs, rewards, dones, forest_state, info = env.step(
                actions_clipped, forest_state
            )

            rollout_rewards.append(rewards)
            rollout_dones.append(dones)

            rollout_min_x.append(info["min_x"])
            rollout_vel_x.append(info["mean_vel_x"])
            rollout_collisions.append(info["collision_rate"])
            rollout_escapes.append(info["escape_rate"])

            # Reset done worlds
            rng_key, reset_key = jax.random.split(rng_key)
            obs, forest_state = env.reset_done_worlds(forest_state, reset_key)

            global_step += tc.n_worlds * tc.n_drones

        # Stack rollout
        rollout_obs = jnp.stack(rollout_obs)         # (T, W, D, obs_dim)
        rollout_actions = jnp.stack(rollout_actions)  # (T, W, D, 3)
        rollout_log_probs = jnp.stack(rollout_log_probs)  # (T, W, D) or (T, W, D)
        rollout_values = jnp.stack(rollout_values)    # (T, W, D) or (T, W)
        rollout_rewards = jnp.stack(rollout_rewards)  # (T, W, D)
        rollout_dones = jnp.stack(rollout_dones)      # (T, W)

        # Bootstrap value for last obs
        if multi_drone:
            last_values = value_fn(train_state.params, obs)  # (W, D)
        else:
            obs_flat = obs.reshape(-1, oc.obs_dim_per_drone)
            last_values_flat = value_fn(train_state.params, obs_flat)
            last_values = last_values_flat.reshape(tc.n_worlds, tc.n_drones)

        # Compute GAE
        advantages, returns = compute_gae(
            rollout_rewards, rollout_values, rollout_dones,
            last_values, tc.gamma, tc.gae_lambda,
        )
        # advantages: (T, W, D)
        # returns: (T, W, D) for single-drone, (T, W) for multi-drone

        # ===== PPO Update =====
        T, W = rollout_obs.shape[:2]
        D = tc.n_drones

        if multi_drone:
            # Batch by (world, timestep) — each sample has all drones
            total_samples = T * W
            flat_obs = rollout_obs.reshape(total_samples, D, -1)      # (B, D, obs_dim)
            flat_actions = rollout_actions.reshape(total_samples, D, 3)  # (B, D, 3)
            flat_log_probs = rollout_log_probs.reshape(total_samples, D)  # (B, D)
            flat_advantages = advantages.reshape(total_samples, D)     # (B, D)
            flat_returns = returns.reshape(total_samples, D)           # (B, D) per-drone
            flat_values = rollout_values.reshape(total_samples, D)     # (B, D) per-drone
        else:
            # Flatten all (T, W, D) to individual samples
            total_samples = T * W * D
            flat_obs = rollout_obs.reshape(total_samples, -1)
            flat_actions = rollout_actions.reshape(total_samples, 3)
            flat_log_probs = rollout_log_probs.reshape(total_samples)
            flat_values = rollout_values.reshape(total_samples)
            flat_advantages = advantages.reshape(total_samples)
            flat_returns = returns.reshape(total_samples)

        for epoch in range(tc.n_epochs):
            rng_key, shuffle_key = jax.random.split(rng_key)
            perm = jax.random.permutation(shuffle_key, total_samples)

            for start in range(0, total_samples, tc.minibatch_size):
                end = min(start + tc.minibatch_size, total_samples)
                idx = perm[start:end]

                batch = {
                    "obs": flat_obs[idx],
                    "actions": flat_actions[idx],
                    "log_probs": flat_log_probs[idx],
                    "values": flat_values[idx],
                    "advantages": flat_advantages[idx],
                    "returns": flat_returns[idx],
                }

                train_state, update_info = update_fn(train_state, batch, current_ent_coef)
                n_updates += 1

        # ===== Logging =====
        elapsed = time.time() - start_time
        fps = global_step / max(elapsed, 1e-6)

        var_y = jnp.var(flat_returns)
        explained_var = 1.0 - jnp.var(flat_returns - flat_values) / (var_y + 1e-8)

        # Action std from policy
        if multi_drone:
            model_tmp = MultiAgentActorCritic(
                encoder_dim=tc.encoder_dim, hidden_dim=tc.critic_hidden_dim,
                act_dim=3, n_heads=tc.n_heads,
            )
            means, log_std = model_tmp.apply(
                train_state.params, flat_obs[0], method=model_tmp.get_action_dist,
            )
            action_std = float(jnp.exp(log_std).mean())
        else:
            model_tmp = ActorCritic(hidden_dims=tc.hidden_dims, act_dim=3, n_drones=tc.n_drones)
            _, log_std, _ = model_tmp.apply(train_state.params, flat_obs[0])
            action_std = float(jnp.exp(log_std).mean())

        mean_min_x = float(jnp.stack(rollout_min_x).mean())
        mean_vel_x = float(jnp.stack(rollout_vel_x).mean())
        mean_collision = float(jnp.stack(rollout_collisions).mean())
        mean_escape = float(jnp.stack(rollout_escapes).mean())

        log_data = {
            "global_step": int(global_step),
            "fps": float(fps),
            "elapsed_sec": float(elapsed),
            "mean_reward": float(rollout_rewards.mean()),
            "mean_min_x": mean_min_x,
            "mean_vel_x": mean_vel_x,
            "collision_rate": mean_collision,
            "escape_rate": mean_escape,
            "policy_loss": float(update_info["policy_loss"]),
            "value_loss": float(update_info["value_loss"]),
            "entropy": float(update_info["entropy"]),
            "action_std": action_std,
            "clip_fraction": float(update_info["clip_fraction"]),
            "approx_kl": float(update_info["approx_kl"]),
            "explained_variance": float(explained_var),
            "n_updates": n_updates,
            "return_mean": float(flat_returns.mean()),
            "return_std": float(flat_returns.std()),
            "advantage_std": float(flat_advantages.std()),
            "n_drones": tc.n_drones,
            "phase": phase,
            "n_active_trees": env.n_active_trees,
            "ent_coef": float(current_ent_coef),
        }

        print(
            f"Step {global_step:>8,} | "
            f"FPS {fps:>8,.0f} | "
            f"R {log_data['mean_reward']:>7.3f} | "
            f"MinX {mean_min_x:>6.3f} | "
            f"VelX {mean_vel_x:>6.3f} | "
            f"Esc {mean_escape:.3f} | "
            f"KL {log_data['approx_kl']:.4f} | "
            f"Std {action_std:.3f} | "
            f"EV {log_data['explained_variance']:.3f} | "
            f"Trees {env.n_active_trees} | "
            f"Ent {current_ent_coef:.4f}"
        )

        with open(log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # Save checkpoint
        if global_step % tc.save_freq < steps_per_rollout:
            ckpt_path = os.path.join(tc.save_dir, f"checkpoint_{global_step}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "params": jax.device_get(train_state.params),
                    "step": global_step,
                    "config": {"tc": tc, "fc": fc, "oc": oc, "rc": rc},
                }, f)
            print(f"  Saved checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\n=== Training complete ===")
    print(f"Total steps: {global_step:,}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Average FPS: {global_step/total_time:,.0f}")

    return train_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ForestEscape with MAPPO")
    parser.add_argument("--n_worlds", type=int, default=1024)
    parser.add_argument("--n_drones", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results/crazyflow")
    parser.add_argument("--lr_init", type=float, default=3e-4)
    parser.add_argument("--lr_final", type=float, default=5e-5)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--encoder_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=2)
    args = parser.parse_args()

    tc = TrainConfig(
        n_worlds=args.n_worlds,
        n_drones=args.n_drones,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        save_dir=args.save_dir,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        encoder_dim=args.encoder_dim,
        n_heads=args.n_heads,
    )

    train(tc)
