# TODO — ForestEscape Next Steps

## Current State

Single-drone navigation through 20-tree forest: **98.4% success rate** (v10, 300M steps on Crazyflow).

## Priority 1: Scale Tree Density

The 20-tree forest is mastered. Test the policy's limits with denser forests.

- [ ] **v11: 25 trees** — Resume from v10 checkpoint with 25 trees. Spacing=0.89m, collision gap=0.59m. Should be navigable but harder. Curriculum 20→25 over 10% of training.
- [ ] **v12: 30 trees** — If 25 works (>80% nav), push to 30 trees. Spacing=0.82m, collision gap=0.52m.
- [ ] **v13: 35 trees** — Original PyBullet density. Spacing=0.76m, collision gap=0.46m. The ultimate single-drone test.

## Priority 2: Multi-Drone with terminate_on_collision

Now that single-drone navigation is solved, add multi-agent coordination.

- [ ] **3-drone MAPPO**: Use AttentionActor + CentralizedCritic from Phase 2 v1–v6 (already implemented in `networks.py`). Start with 20 trees, terminate_on_collision=True.
- [ ] **Credit assignment**: With terminate_on_collision, any drone colliding ends the episode for all. Need to balance team vs individual incentives.
- [ ] **Curriculum**: Start with 1 drone (known-good), gradually add drones? Or start with 3 drones but very few trees?

## Priority 3: Architecture Improvements

- [ ] **Recurrent policy (GRU/LSTM)**: For sequential navigation decisions — the MLP has no memory of past tree encounters. May help with the remaining 1.6% failure cases.
- [ ] **Larger network**: (1024, 512) MLP for denser forests where more capacity is needed.
- [ ] **Depth vision**: Replace synthetic lidar with `mjx.ray()` GPU ray-casting for more realistic observations.

## Priority 4: Evaluation & Robustness

- [ ] **Domain randomization**: Vary tree count (15–35), radius (0.04–0.08m), forest layout during training.
- [ ] **Zero-shot generalization**: Test v10 policy on unseen tree counts (25, 30, 35) without retraining.
- [ ] **Failure analysis**: Visualize the 1.6% failure cases — are they truly impossible configurations or fixable?

## Notes

- Training server: `ai@172.31.10.232`, `.venv/bin/python`
- Best checkpoint: `results/crazyflow_v10_continued/` (latest ~300M step checkpoint)
- v9 checkpoint (100M, 68% nav): `results/crazyflow_v9_realcoll/checkpoint_100663296.pkl`
- All Crazyflow code: `gym_pybullet_drones/crazyflow/`
