"""Short headless check of the same environment used by training."""

from __future__ import annotations

import argparse
import json

import torch

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.task.vec_env import TaskVecEnv


def run(config: ResolvedRunConfig, args: argparse.Namespace) -> int:
    with TaskVecEnv(config) as environment:
        observations, _ = environment.reset()
        actions = torch.zeros((environment.num_envs, environment.num_actions), dtype=torch.float32)
        completed_episodes = 0
        for _ in range(args.steps):
            observations, rewards, dones, _ = environment.step(actions)
            if not torch.isfinite(observations).all() or not torch.isfinite(rewards).all():
                raise RuntimeError("smoke check encountered non-finite observations or rewards")
            completed_episodes += int(dones.sum())
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "num_envs": environment.num_envs,
                    "steps": args.steps,
                    "observation_shape": list(observations.shape),
                    "completed_episodes": completed_episodes,
                },
                indent=2,
            )
        )
    return 0
