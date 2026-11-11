"""Training and checkpoint runner for the reach policy's feed-forward PPO."""

from __future__ import annotations

import os
import random
import statistics
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration as HH:MM:SS."""

    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_scalar(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1.0e4 or (0.0 < absolute < 1.0e-4):
        return f"{value:.3e}"
    return f"{value:.6f}"


def _capture_rng_state() -> dict[str, object]:
    """Capture process RNGs needed for reproducible iteration-boundary resume."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    """Restore a state written by :func:`_capture_rng_state`."""

    required = ("python", "numpy", "torch_cpu", "torch_cuda")
    missing = [name for name in required if name not in state]
    if missing:
        raise RuntimeError(f"Runner RNG checkpoint is incomplete: missing={missing}")
    random.setstate(cast(tuple[Any, ...], state["python"]))
    np.random.set_state(cast(Any, state["numpy"]))
    torch_cpu_state = state["torch_cpu"]
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise RuntimeError("Checkpoint Torch CPU RNG state must be a tensor")
    torch.set_rng_state(torch_cpu_state.to(device="cpu"))
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        if not isinstance(cuda_state, Sequence) or isinstance(cuda_state, (str, bytes)):
            raise RuntimeError("Checkpoint CUDA RNG state must be a tensor sequence")
        torch.cuda.set_rng_state_all([torch.as_tensor(value, device="cpu") for value in cuda_state])


def _validate_checkpoint_action_space(
    checkpoint_action_space: object,
    expected_action_space: dict[str, object],
) -> None:
    """Require the checkpoint's exact bounded-action contract."""

    if checkpoint_action_space != expected_action_space:
        raise RuntimeError(
            "Checkpoint policy action-space mismatch: "
            f"checkpoint={checkpoint_action_space}, "
            f"configured={expected_action_space}"
        )


def _training_progress_text(
    *,
    iteration: int,
    final_iteration: int,
    fps: int,
    collection_time: float,
    learning_time: float,
    losses: dict[str, float],
    mean_reward: float | None,
    mean_episode_length: float | None,
    episode_metrics: dict[str, float],
    total_timesteps: int,
    total_time: float,
    eta_seconds: float,
    algorithm_update_count: int,
    critic_warmup_updates: int,
) -> str:
    """Build the human-readable per-iteration terminal progress panel."""

    width = 88
    lines = [
        "#" * width,
        f"{f'Learning iteration {iteration + 1}/{final_iteration}':^{width}}",
        "",
        (f"{'Computation':>42}: {fps} steps/s (collection: {collection_time:.3f}s, learning: {learning_time:.3f}s)"),
    ]
    loss_labels = {
        "value_function": "Mean value function loss",
        "surrogate": "Mean surrogate loss",
        "entropy": "Mean entropy",
        "actor_anchor": "Mean actor anchor loss",
    }
    for name, label in loss_labels.items():
        if name in losses:
            lines.append(f"{label:>42}: {_format_scalar(float(losses[name]))}")

    actor_enabled = bool(losses.get("actor_update_enabled", 1.0))
    if actor_enabled:
        actor_status = "active"
    else:
        warmup_progress = min(algorithm_update_count, critic_warmup_updates)
        actor_status = f"critic warmup ({warmup_progress}/{critic_warmup_updates})"
    lines.append(f"{'Actor update':>42}: {actor_status}")
    if "actor_learning_rate" in losses:
        lines.append(f"{'Actor learning rate':>42}: {float(losses['actor_learning_rate']):.3e}")
    if "critic_learning_rate" in losses:
        lines.append(f"{'Critic learning rate':>42}: {float(losses['critic_learning_rate']):.3e}")
    if "actor_anchor_weight" in losses:
        lines.append(f"{'Actor anchor weight':>42}: {_format_scalar(float(losses['actor_anchor_weight']))}")
    if mean_reward is not None:
        lines.append(f"{'Mean reward':>42}: {_format_scalar(mean_reward)}")
    if mean_episode_length is not None:
        lines.append(f"{'Mean episode length':>42}: {_format_scalar(mean_episode_length)}")

    if episode_metrics:
        lines.extend(("", "Rollout environment metrics:"))
        for name, value in episode_metrics.items():
            label = name.removeprefix("/")
            lines.append(f"{label:>42}: {_format_scalar(value)}")

    iteration_time = collection_time + learning_time
    lines.extend(
        (
            "-" * width,
            f"{'Total timesteps':>42}: {total_timesteps}",
            f"{'Iteration time':>42}: {iteration_time:.2f}s",
            f"{'Time elapsed':>42}: {_format_duration(total_time)}",
            f"{'ETA':>42}: {_format_duration(eta_seconds)}",
        )
    )
    return "\n".join(lines)


class OnPolicyRunner:
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = train_cfg
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env
        self._configure_multi_gpu()

        observations, extras = env.get_observations()
        critic_observations = extras["observations"].get("critic", observations)
        self.privileged_obs_type = "critic" if "critic" in extras["observations"] else None
        policy = ActorCritic(
            observations.shape[1],
            critic_observations.shape[1],
            env.num_actions,
            **self.policy_cfg,
        ).to(device)
        self.alg = PPO(
            policy,
            device=device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]
        self.save_final_only = False
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [observations.shape[1]],
            [critic_observations.shape[1]],
            [env.num_actions],
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer: SummaryWriter | None = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        probe_count = min(32, observations.shape[0])
        self.deterministic_probe_observations = observations[:probe_count].detach().clone().to(self.device)
        probe_targets = getattr(env, "scratch_curriculum_target_xy", None)
        self.deterministic_probe_target_xy = (
            probe_targets[:probe_count].detach().clone().cpu() if isinstance(probe_targets, torch.Tensor) else None
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        observations, extras = self.env.get_observations()
        critic_observations = extras["observations"].get(self.privileged_obs_type, observations)
        observations = observations.to(self.device)
        critic_observations = critic_observations.to(self.device)
        self.train_mode()

        episode_infos: list[dict] = []
        reward_buffer: deque[float] = deque(maxlen=100)
        length_buffer: deque[float] = deque(maxlen=100)
        current_rewards = torch.zeros(self.env.num_envs, device=self.device)
        current_lengths = torch.zeros(self.env.num_envs, device=self.device)

        if self.is_distributed:
            self.alg.broadcast_parameters()

        start_iteration = self.current_learning_iteration
        final_iteration = start_iteration + num_learning_iterations
        for iteration in range(start_iteration, final_iteration):
            collection_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(observations, critic_observations)
                    observations, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    observations = observations.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    critic_observations = (
                        infos["observations"].get(self.privileged_obs_type, observations).to(self.device)
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        info = infos.get("episode", infos.get("log"))
                        if info is not None:
                            episode_infos.append(info)
                        current_rewards += rewards
                        current_lengths += 1
                        completed = (dones > 0).nonzero(as_tuple=False)[:, 0]
                        reward_buffer.extend(current_rewards[completed].cpu().tolist())
                        length_buffer.extend(current_lengths[completed].cpu().tolist())
                        current_rewards[completed] = 0
                        current_lengths[completed] = 0

                collection_time = time.time() - collection_start
                self.alg.compute_returns(critic_observations)

            learning_start = time.time()
            losses = self.alg.update()
            learning_time = time.time() - learning_start
            self.current_learning_iteration = iteration

            if self.writer is not None:
                if self.log_dir is None:
                    raise RuntimeError("SummaryWriter requires a log directory")
                self._log(
                    iteration,
                    start_iteration,
                    final_iteration,
                    losses,
                    collection_time,
                    learning_time,
                    episode_infos,
                    reward_buffer,
                    length_buffer,
                )
                if not self.save_final_only and iteration % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
            episode_infos.clear()

        if self.writer is not None:
            if self.log_dir is None:
                raise RuntimeError("SummaryWriter requires a log directory")
            self.save(
                os.path.join(
                    self.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )

    def _log(
        self,
        iteration: int,
        start_iteration: int,
        final_iteration: int,
        losses: dict[str, float],
        collection_time: float,
        learning_time: float,
        episode_infos: list[dict],
        reward_buffer: deque[float],
        length_buffer: deque[float],
    ) -> None:
        if self.writer is None:
            raise RuntimeError("logging requires an initialized SummaryWriter")
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        iteration_time = collection_time + learning_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time
        fps = int(collection_size / max(iteration_time, 1.0e-9))

        for name, value in losses.items():
            self.writer.add_scalar(f"Loss/{name}", value, iteration)
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
        self.writer.add_scalar("Perf/total_fps", fps, iteration)
        mean_reward = None
        mean_episode_length = None
        if reward_buffer:
            mean_reward = statistics.mean(reward_buffer)
            mean_episode_length = statistics.mean(length_buffer)
            self.writer.add_scalar(
                "Train/mean_reward",
                mean_reward,
                iteration,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                mean_episode_length,
                iteration,
            )
        episode_metrics: dict[str, float] = {}
        for name in episode_infos[0] if episode_infos else ():
            values = [info[name] for info in episode_infos if name in info]
            tensors = [torch.as_tensor(value, device=self.device) for value in values]
            if tensors:
                if name.startswith(("curriculum/", "option/")):
                    # Curriculum and Option entries are accumulator snapshots,
                    # not independent timestep samples. Report the latest
                    # state instead of averaging old and new counters.
                    mean_value = float(tensors[-1].float().mean())
                else:
                    mean_value = float(torch.cat([value.reshape(-1) for value in tensors]).mean())
                episode_metrics[name] = mean_value
                self.writer.add_scalar(
                    name if "/" in name else f"Episode/{name}",
                    mean_value,
                    iteration,
                )
        with torch.inference_mode():
            deterministic_probe_actions = self.alg.policy.act_inference(self.deterministic_probe_observations)
            deterministic_probe_rms = float(torch.sqrt(torch.mean(torch.square(deterministic_probe_actions))))
            noise_std_mean = float(self.alg.policy._effective_std().mean())
        policy_diagnostics = {
            "policy/deterministic_action_rms_on_probe_set": (deterministic_probe_rms),
            "policy/noise_std_mean": noise_std_mean,
        }
        episode_metrics.update(policy_diagnostics)
        for name, value in policy_diagnostics.items():
            self.writer.add_scalar(name, value, iteration)

        completed_iterations = iteration - start_iteration + 1
        mean_iteration_time = self.tot_time / max(completed_iterations, 1)
        remaining_iterations = max(final_iteration - iteration - 1, 0)
        eta_seconds = mean_iteration_time * remaining_iterations
        print(
            _training_progress_text(
                iteration=iteration,
                final_iteration=final_iteration,
                fps=fps,
                collection_time=collection_time,
                learning_time=learning_time,
                losses=losses,
                mean_reward=mean_reward,
                mean_episode_length=mean_episode_length,
                episode_metrics=episode_metrics,
                total_timesteps=self.tot_timesteps,
                total_time=self.tot_time,
                eta_seconds=eta_seconds,
                algorithm_update_count=self.alg.update_count,
                critic_warmup_updates=self.alg.critic_warmup_updates,
            ),
            flush=True,
        )

    def save(self, path: str, infos: object | None = None) -> None:
        checkpoint = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "algorithm_state_dict": self.alg.get_checkpoint_state(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "rng_state_dict": _capture_rng_state(),
            "policy_action_space": self.alg.policy.action_space_metadata(),
        }
        option_configuration = getattr(self.env, "policy_option_configuration", None)
        if option_configuration is not None:
            checkpoint["policy_option_configuration"] = dict(option_configuration)
        unsafe_displacement_tail_configuration = getattr(
            self.env,
            "unsafe_displacement_tail_configuration",
            None,
        )
        if unsafe_displacement_tail_configuration is not None:
            checkpoint["unsafe_displacement_tail_configuration"] = dict(unsafe_displacement_tail_configuration)
        if hasattr(self.env, "get_checkpoint_state"):
            checkpoint["env_state_dict"] = self.env.get_checkpoint_state()
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = True) -> object:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        expected_action_space = self.alg.policy.action_space_metadata()
        checkpoint_action_space = checkpoint.get("policy_action_space")
        _validate_checkpoint_action_space(checkpoint_action_space, expected_action_space)
        resumed_training = self.alg.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if resumed_training:
            checkpoint_iteration = int(checkpoint.get("iter", 0))
            algorithm_state = checkpoint.get("algorithm_state_dict")
            if isinstance(algorithm_state, dict):
                self.alg.load_checkpoint_state(algorithm_state)
                # ``update_count`` is the next zero-based learning iteration.
                # A model_19 checkpoint has already completed 20 updates, so
                # resume at iteration 20 rather than repeating iteration 19.
                self.current_learning_iteration = self.alg.update_count
            else:
                # Legacy checkpoints did not persist warmup/anchor state.
                # Treat the loaded Actor as the new conservative reference and
                # continue after the number of updates represented by ``iter``.
                self.current_learning_iteration = checkpoint_iteration + 1
                self.alg.update_count = self.current_learning_iteration
                if self.alg.actor_anchor_weight > 0.0:
                    self.alg.set_actor_anchor()
        rng_state = checkpoint.get("rng_state_dict")
        if isinstance(rng_state, dict):
            _restore_rng_state(rng_state)
        if "env_state_dict" in checkpoint and hasattr(self.env, "load_checkpoint_state"):
            self.env.load_checkpoint_state(checkpoint["env_state_dict"])
        elif getattr(self.env, "requires_checkpoint_state", False):
            raise RuntimeError(
                "This environment requires checkpoint state, but the loaded checkpoint does not contain env_state_dict"
            )
        return checkpoint.get("infos")

    def get_inference_policy(self, device: str | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        self.alg.policy.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device {self.device!r} does not match local rank {self.gpu_local_rank}")
        if not 0 <= self.gpu_local_rank < self.gpu_world_size:
            raise ValueError("Local rank is outside the distributed world")
        if not 0 <= self.gpu_global_rank < self.gpu_world_size:
            raise ValueError("Global rank is outside the distributed world")
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.gpu_global_rank,
            world_size=self.gpu_world_size,
        )
        torch.cuda.set_device(self.gpu_local_rank)
