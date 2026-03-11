"""Quick diagnostic test for training pipeline."""

import jax
import jax.numpy as jnp
from .config import TrainConfig, ObsConfig
from .forest_escape_env import ForestEscapeEnv
from .networks import ActorCritic
from .train_mappo import create_train_state

def main():
    print("GPU:", jax.devices())

    tc = TrainConfig(n_worlds=16, n_drones=1)
    oc = ObsConfig()
    env = ForestEscapeEnv(tc)
    obs, fs = env.reset(jax.random.PRNGKey(0))

    print("Obs shape:", obs.shape)
    print("Obs stats: mean={:.3f} std={:.3f} min={:.3f} max={:.3f}".format(
        obs.mean().item(), obs.std().item(), obs.min().item(), obs.max().item()))

    # Rollout with random actions
    for i in range(30):
        actions = jax.random.uniform(jax.random.PRNGKey(i), (16, 1, 3), minval=-1, maxval=1)
        obs, rewards, dones, fs, info = env.step(actions, fs)
        if i < 5 or i >= 25:
            r = rewards.mean().item()
            vel = info["mean_vel_x"].item()
            mx = info["min_x"].item()
            print(f"  Step {i:2d}: reward={r:7.3f} vel_x={vel:.4f} min_x={mx:.3f}")

    print("\nReward at step 29:", rewards.mean().item())

    # Check value function initial scale
    ts = create_train_state(jax.random.PRNGKey(0), tc, oc)
    model = ActorCritic(hidden_dims=(256, 256), act_dim=3, n_drones=1)
    obs_flat = obs.reshape(-1, 60)
    values = jax.vmap(lambda o: model.apply(ts.params, o)[2])(obs_flat)
    print("\nInitial value predictions: mean={:.4f} std={:.4f}".format(
        values.mean().item(), values.std().item()))

    # Check gradient flow
    def dummy_loss(params):
        log_probs, values, entropy = jax.vmap(
            lambda o, a: model.apply(params, o, a, method=model.evaluate_action)
        )(obs_flat, jnp.zeros((obs_flat.shape[0], 3)))
        return (values ** 2).mean() + log_probs.mean()

    grads = jax.grad(dummy_loss)(ts.params)
    grad_norms = {k: jnp.linalg.norm(jax.tree.leaves(v)[0]).item()
                  for k, v in grads["params"].items() if jax.tree.leaves(v)}
    print("\nGradient norms per layer:")
    for k, v in grad_norms.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== Diagnostics complete ===")


if __name__ == "__main__":
    main()
