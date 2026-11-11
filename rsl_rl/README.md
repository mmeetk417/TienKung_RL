# Local RSL-RL

This vendored package provides the feed-forward ActorCritic, bounded tanh-normal
policy, PPO, rollout storage, and vector-environment interface used by TienKung RL.
The current task uses 106 observations and 7 continuous arm actions; the task
runtime triggers the fixed grasp option automatically.

The command-line training loop lives in `tienkung_rl/learning/runner.py`.
The independent runner under `rsl_rl/runners/` also has regression coverage.

The code is derived from RSL-RL and remains under the BSD-3-Clause license in
`LICENSE`.
