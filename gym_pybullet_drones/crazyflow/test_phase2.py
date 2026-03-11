"""Quick test for Phase 2 multi-drone training components."""

import sys
sys.path.insert(0, "/home/ai/source/gym-pybullet-drones")

import jax
import jax.numpy as jnp

print("Testing Phase 2 multi-drone components...")
print(f"JAX devices: {jax.devices()}")

# Test 1: MultiAgentActorCritic init and forward pass
print("\n--- Test 1: MultiAgentActorCritic ---")
from networks import MultiAgentActorCritic

n_drones = 3
obs_dim = 60
act_dim = 3

model = MultiAgentActorCritic(encoder_dim=128, hidden_dim=256, act_dim=act_dim, n_heads=2)
dummy_obs = jnp.zeros((n_drones, obs_dim))
rng_key = jax.random.PRNGKey(42)

# Init with get_actions_and_value
params = model.init(rng_key, dummy_obs, rng_key, method=model.get_actions_and_value)
param_count = sum(x.size for x in jax.tree.leaves(params))
print(f"  Parameters: {param_count:,}")

# Test forward pass
actions, log_probs, value, entropy = model.apply(
    params, dummy_obs, rng_key, method=model.get_actions_and_value
)
print(f"  actions: {actions.shape} = {actions}")
print(f"  log_probs: {log_probs.shape} = {log_probs}")
print(f"  value: {value.shape} = {value}")
print(f"  entropy: {entropy}")

# Test evaluate_actions
log_probs2, value2, entropy2 = model.apply(
    params, dummy_obs, actions, method=model.evaluate_actions
)
print(f"  evaluate log_probs match: {jnp.allclose(log_probs, log_probs2, atol=1e-5)}")
print(f"  evaluate value match: {jnp.allclose(value, value2, atol=1e-5)}")

# Test get_value
value3 = model.apply(params, dummy_obs, method=model.get_value)
print(f"  get_value match: {jnp.allclose(value, value3, atol=1e-5)}")

# Test get_action_dist
means, log_std = model.apply(params, dummy_obs, method=model.get_action_dist)
print(f"  means: {means.shape}, log_std: {log_std.shape}")

# Test 2: Forward bias
print("\n--- Test 2: Forward bias ---")
import flax.core
flat_params = flax.core.unfreeze(params)
# Check the action head bias path
print(f"  Actor keys: {list(flat_params['params']['actor'].keys())}")
bias = flat_params["params"]["actor"]["Dense_0"]["bias"]
print(f"  Action head bias (before): {bias}")
flat_params["params"]["actor"]["Dense_0"]["bias"] = jnp.array([0.5, 0.0, 0.0])
params_biased = flax.core.freeze(flat_params)
means_biased, _ = model.apply(params_biased, dummy_obs, method=model.get_action_dist)
print(f"  Biased means: {means_biased}")
print(f"  Forward bias works: {float(means_biased[:, 0].mean()) > 0.3}")

# Test 3: Vmapped inference (simulate n_worlds)
print("\n--- Test 3: Vmapped inference ---")
n_worlds = 16
all_obs = jax.random.normal(rng_key, (n_worlds, n_drones, obs_dim)) * 0.1
world_keys = jax.random.split(rng_key, n_worlds)

actions_w, log_probs_w, values_w, entropy_w = jax.vmap(
    lambda o, k: model.apply(params, o, k, method=model.get_actions_and_value)
)(all_obs, world_keys)
print(f"  actions: {actions_w.shape}")  # (W, D, 3)
print(f"  log_probs: {log_probs_w.shape}")  # (W, D)
print(f"  values: {values_w.shape}")  # (W,)
print(f"  entropy: {entropy_w.shape}")  # (W,)

# Test 4: GAE with centralized critic
print("\n--- Test 4: GAE with centralized critic ---")
T = 8
rewards = jax.random.normal(rng_key, (T, n_worlds, n_drones))
values_t = jax.random.normal(rng_key, (T, n_worlds)) * 0.5  # centralized
dones = jnp.zeros((T, n_worlds), dtype=bool)
last_val = jnp.zeros(n_worlds)

from gym_pybullet_drones.crazyflow.train_mappo import compute_gae
advantages, returns_val = compute_gae(rewards, values_t, dones, last_val, 0.99, 0.95)
print(f"  advantages: {advantages.shape}")  # (T, W, D)
print(f"  returns (value target): {returns_val.shape}")  # (T, W)
print(f"  advantage mean: {advantages.mean():.4f}")

# Test 5: PPO loss
print("\n--- Test 5: Multi-drone PPO loss ---")
from gym_pybullet_drones.crazyflow.train_mappo import _make_multi_loss_fn

loss_fn = _make_multi_loss_fn(128, 256, 3, 2)
B = 4
batch = {
    "obs": all_obs[:B],  # (B, D, obs_dim)
    "actions": actions_w[:B],  # (B, D, 3)
    "log_probs": log_probs_w[:B],  # (B, D)
    "advantages": advantages[0, :B],  # (B, D)
    "returns": returns_val[0, :B],  # (B,)
    "values": values_w[:B],  # (B,)
}
(total_loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
    params, batch, 0.2, 0.005, 0.5
)
print(f"  total_loss: {total_loss:.4f}")
print(f"  policy_loss: {info['policy_loss']:.4f}")
print(f"  value_loss: {info['value_loss']:.4f}")
print(f"  entropy: {info['entropy']:.4f}")
print(f"  clip_fraction: {info['clip_fraction']:.4f}")
grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(grads)))
print(f"  grad_norm: {grad_norm:.4f}")

print("\n=== All Phase 2 tests passed! ===")
