"""Finite, reproducible CPU training orchestration for reach-then-close."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import canonical_json
from tienkung_rl.task.environment import TaskStepTransition
from tienkung_rl.task.vec_env import TaskVecEnv
from tienkung_rl.telemetry.metrics import (
    CurriculumWindowEvent,
    EpisodeEvent,
    PPOUpdateEvent,
    RunEvent,
    StepEvent,
    TelemetryAggregator,
)
from tienkung_rl.telemetry.reports import (
    ReportMetadata,
    finalize_run_manifest,
    prepare_run_directory,
    render_terminal_summary,
    write_final_reports,
    write_json_report,
)
from tienkung_rl.telemetry.tensorboard import TensorBoardTelemetryWriter

from .checkpoint import (
    EPISODE_BOUNDARY_SEMANTICS,
    CheckpointError,
    ValidatedCheckpoint,
    build_sidecar_metadata,
    checkpoint_name,
    dependency_snapshot,
    load_checkpoint_preflight,
    prune_periodic_checkpoints,
    save_checkpoint,
)

RUNNER_STATE_VERSION: Final = "tienkung_cpu_training_runner_v2"
RNG_STATE_VERSION: Final = "python_numpy_torch_cpu_rng_v1"


@dataclass(slots=True)
class RunnerState:
    """Iteration-boundary state that remains meaningful after episode discard."""

    version: str = RUNNER_STATE_VERSION
    completed_iterations: int = 0
    total_policy_steps: int = 0
    total_physics_steps: int = 0
    completed_episodes: int = 0
    sampling_seconds: float = 0.0
    learning_seconds: float = 0.0
    total_seconds: float = 0.0
    last_losses: dict[str, float] | None = None
    last_actor_gradient_norm: float = 0.0

    def validate(self) -> None:
        if self.version != RUNNER_STATE_VERSION:
            raise CheckpointError("unknown runner state version")
        for name in (
            "completed_iterations",
            "total_policy_steps",
            "total_physics_steps",
            "completed_episodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CheckpointError(f"runner state {name} must be a non-negative integer")
        for name in (
            "sampling_seconds",
            "learning_seconds",
            "total_seconds",
            "last_actor_gradient_norm",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise CheckpointError(f"runner state {name} must be finite and non-negative")
        if self.last_losses is not None and (
            not isinstance(self.last_losses, dict)
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in self.last_losses.values()
            )
        ):
            raise CheckpointError("runner state losses must be a finite mapping")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RunnerState:
        required = {
            "version",
            "completed_iterations",
            "total_policy_steps",
            "total_physics_steps",
            "completed_episodes",
            "sampling_seconds",
            "learning_seconds",
            "total_seconds",
            "last_losses",
            "last_actor_gradient_norm",
        }
        if set(raw) != required:
            raise CheckpointError("runner state fields are incomplete or unknown")
        losses = raw["last_losses"]
        if losses is not None and not isinstance(losses, Mapping):
            raise CheckpointError("runner state last_losses must be a mapping or null")
        state = cls(
            version=cast(str, raw["version"]),
            completed_iterations=cast(int, raw["completed_iterations"]),
            total_policy_steps=cast(int, raw["total_policy_steps"]),
            total_physics_steps=cast(int, raw["total_physics_steps"]),
            completed_episodes=cast(int, raw["completed_episodes"]),
            sampling_seconds=float(cast(float, raw["sampling_seconds"])),
            learning_seconds=float(cast(float, raw["learning_seconds"])),
            total_seconds=float(cast(float, raw["total_seconds"])),
            last_losses=None if losses is None else {str(key): float(value) for key, value in losses.items()},
            last_actor_gradient_norm=float(cast(float, raw["last_actor_gradient_norm"])),
        )
        state.validate()
        return state


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    policy: ActorCritic
    ppo: PPO


def effective_config(config: ResolvedRunConfig, args: argparse.Namespace) -> ResolvedRunConfig:
    """Apply train-only CLI values before any fingerprint or snapshot is made."""

    maximum_iterations = getattr(args, "max_iterations", None)
    save_interval = getattr(args, "save_interval", None)
    run_config = replace(
        config.run,
        maximum_iterations=(config.run.maximum_iterations if maximum_iterations is None else maximum_iterations),
        save_interval_iterations=(config.run.save_interval_iterations if save_interval is None else save_interval),
    )
    resolved = replace(config, run=run_config)
    resolved.validate()
    return resolved


def seed_all(seed: int) -> None:
    """Seed only the supported Python, NumPy, and Torch CPU processes."""

    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must lie in [0, 2^63-1]")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def capture_process_rng_state() -> dict[str, object]:
    return {
        "version": RNG_STATE_VERSION,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda": None,
    }


def restore_process_rng_state(state: Mapping[str, object]) -> None:
    required = {"version", "python", "numpy", "torch_cpu", "cuda"}
    if set(state) != required:
        raise CheckpointError("process RNG checkpoint fields are incomplete or unknown")
    if state["version"] != RNG_STATE_VERSION:
        raise CheckpointError("unknown process RNG checkpoint version")
    if state["cuda"] is not None:
        raise CheckpointError("CUDA RNG state is forbidden by the CPU-only training contract")
    try:
        random.setstate(cast(tuple[Any, ...], state["python"]))
        np.random.set_state(cast(Any, state["numpy"]))
        torch_state = state["torch_cpu"]
        if not isinstance(torch_state, torch.Tensor) or torch_state.device.type != "cpu":
            raise TypeError("Torch RNG state must be a CPU tensor")
        torch.set_rng_state(torch_state)
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"invalid process RNG checkpoint: {error}") from error


def build_training_components(config: ResolvedRunConfig) -> TrainingComponents:
    """Randomly initialize the 106-to-7 arm policy and PPO."""

    config.validate()
    network = config.ppo.network
    policy = ActorCritic(
        num_actor_obs=106,
        num_critic_obs=106,
        num_actions=7,
        actor_hidden_dims=network.actor_hidden_dimensions,
        critic_hidden_dims=network.critic_hidden_dimensions,
        activation=network.activation,
        init_noise_std=config.ppo.initial_continuous_std,
        min_std=config.ppo.minimum_continuous_std,
        max_std=config.ppo.maximum_continuous_std,
    )
    ppo = PPO(
        policy,
        num_learning_epochs=config.ppo.learning_epochs,
        num_mini_batches=config.ppo.mini_batches,
        clip_param=config.ppo.clip_ratio,
        gamma=config.ppo.gamma,
        lam=config.ppo.gae_lambda,
        value_loss_coef=config.ppo.value_loss_coefficient,
        entropy_coef=config.ppo.entropy_coefficient,
        actor_learning_rate=config.ppo.actor_learning_rate,
        critic_learning_rate=config.ppo.critic_learning_rate,
        max_grad_norm=config.ppo.maximum_gradient_norm,
        desired_kl=config.ppo.desired_kl,
        schedule="adaptive",
        device="cpu",
    )
    ppo.init_storage(
        config.platform.num_envs,
        config.ppo.rollout_steps_per_env,
        [106],
        [106],
        [7],
    )
    return TrainingComponents(policy, ppo)


def _storage(ppo: PPO) -> RolloutStorage:
    if ppo.storage is None:
        raise RuntimeError("PPO storage has not been initialized")
    return ppo.storage


def _finite_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...] | None = None) -> None:
    if shape is not None and tuple(value.shape) != shape:
        raise RuntimeError(f"{name} shape must be {shape}, got {tuple(value.shape)}")
    if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} must be a finite CPU tensor")


def terminal_bootstrap_values(
    policy: ActorCritic,
    transitions: Sequence[TaskStepTransition],
) -> torch.Tensor:
    """Evaluate the real pre-reset terminal observations as one critic batch."""

    if not transitions:
        raise ValueError("terminal transition batch must not be empty")
    terminal_observations = torch.stack(
        [transition.terminal_observation.values for transition in transitions],
        dim=0,
    ).to(dtype=torch.float32, device="cpu")
    _finite_tensor("terminal observations", terminal_observations, (len(transitions), 106))
    with torch.inference_mode():
        values = policy.evaluate(terminal_observations).detach().clone()
    _finite_tensor("terminal bootstrap values", values, (len(transitions), 1))
    return values


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            _finite_tensor("optimizer gradient", parameter.grad)
            squares.append(float(torch.sum(parameter.grad.detach().square()).item()))
    return math.sqrt(math.fsum(squares))


def gradient_diagnostics(policy: ActorCritic) -> float:
    return _gradient_norm([*policy.actor.parameters(), policy.std])


def _finite_optimizer_state(ppo: PPO) -> None:
    for parameter in ppo.policy.parameters():
        _finite_tensor("model parameter", parameter)
    for state in ppo.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                _finite_tensor("optimizer state", value)


def _step_event(
    iteration: int,
    action: torch.Tensor,
    transition: TaskStepTransition,
) -> StepEvent:
    snapshot = transition.snapshot
    metrics = transition.admission.metrics
    continuous = tuple(float(value) for value in action[:7].tolist())
    return StepEvent(
        iteration=iteration,
        env_id=transition.env_id,
        policy_step=snapshot.episode_step,
        physics_steps=4,
        actions=continuous,
        action_clipped=tuple(abs(value) >= 1.0 for value in continuous),
        option_admissible=transition.admission.option_admissible,
        trigger_requested=snapshot.trigger_requested,
        trigger_accepted=snapshot.trigger_accepted,
        trigger_rejected=snapshot.trigger_rejected,
        strict_ready=transition.ready.ready,
        admission_failures=transition.admission.failure_reasons,
        accepted_position_error_m=(metrics.position_error_m if snapshot.trigger_accepted else None),
        accepted_orientation_error_rad=(metrics.orientation_error_rad if snapshot.trigger_accepted else None),
        maximum_contact_impulse_n_s=snapshot.illegal_contact_impulse_n_s,
        object_tilt_rad=snapshot.object_tilt_rad,
        object_displacement_m=snapshot.object_displacement_m,
    )


def _episode_event(iteration: int, transition: TaskStepTransition) -> EpisodeEvent:
    snapshot = transition.snapshot
    cylinder = snapshot.randomization["cylinder"]
    if not isinstance(cylinder, Mapping):
        raise RuntimeError("terminal randomization snapshot has no cylinder mapping")
    randomization = {
        "cylinder_radius_m": float(cylinder["radius_m"]),
        "cylinder_height_m": float(cylinder["height_m"]),
        "cylinder_mass_kg": float(cylinder["mass_kg"]),
        "hand_sliding_friction": float(cylinder["hand_sliding_friction"]),
        "table_sliding_friction": float(cylinder["table_sliding_friction"]),
    }
    reward_totals = {name: float(value) for name, value in snapshot.reward_terms.items()}
    reward_totals["total"] = snapshot.episode_reward_total
    return EpisodeEvent(
        iteration=iteration,
        env_id=transition.env_id,
        episode_length_steps=snapshot.episode_step,
        stage_at_reset=snapshot.stage_at_reset,
        strict_ready_ever=snapshot.strict_ready_ever,
        lift_5cm=snapshot.lift_5cm,
        complete_success=snapshot.success,
        dangerous=snapshot.unsafe,
        primary_termination_reason=snapshot.primary_termination_reason,
        auxiliary_termination_reasons=snapshot.secondary_termination_reasons,
        reward_totals=reward_totals,
        randomization_samples=randomization,
    )


def collect_rollout(
    environment: TaskVecEnv,
    ppo: PPO,
    observations: torch.Tensor,
    *,
    iteration: int,
    telemetry: TelemetryAggregator,
) -> torch.Tensor:
    """Collect exactly one fixed rollout and write every full M14 transition."""

    num_envs = environment.num_envs
    rollout_steps = environment.config.ppo.rollout_steps_per_env
    for _ in range(rollout_steps):
        _finite_tensor("policy observations", observations, (num_envs, 106))
        with torch.inference_mode():
            actions = ppo.act(observations, observations)
        _finite_tensor("policy actions", actions, (num_envs, 7))
        next_observations, rewards, dones, extras = environment.step(actions)
        _finite_tensor("environment rewards", rewards, (num_envs,))
        _finite_tensor("next observations", next_observations, (num_envs, 106))
        transitions = extras.get("transitions")
        if (
            not isinstance(transitions, tuple)
            or len(transitions) != num_envs
            or any(not isinstance(item, TaskStepTransition) for item in transitions)
        ):
            raise RuntimeError("TaskVecEnv did not provide one ordered TaskStepTransition per environment")
        terminal_values = terminal_bootstrap_values(ppo.policy, transitions)
        infos = dict(extras)
        infos["timeout_bootstrap_values"] = terminal_values
        infos["option_phase"] = tuple(item.snapshot.option_phase for item in transitions)
        ppo.process_env_step(rewards, dones, infos)
        for env_id, transition in enumerate(transitions):
            telemetry.record_step(
                _step_event(
                    iteration,
                    actions[env_id],
                    transition,
                )
            )
            if transition.done:
                telemetry.record_episode(_episode_event(iteration, transition))
                if transition.curriculum_update is not None and transition.curriculum_update.target_episodes > 0:
                    update = transition.curriculum_update
                    telemetry.record_window(
                        CurriculumWindowEvent(
                            iteration=iteration,
                            stage=update.stage.value,
                            completed_episodes=update.completed_episodes,
                            target_episodes=update.target_episodes,
                            promoted=update.promoted,
                            stalled=update.stalled,
                            consecutive_passing_windows=update.consecutive_passing_windows,
                        )
                    )
        observations = next_observations
    ppo.compute_returns(observations)
    storage = _storage(ppo)
    _finite_tensor("rollout returns", storage.returns)
    _finite_tensor("rollout advantages", storage.advantages)
    return observations


def _update_run_manifest(
    run_dir: Path,
    *,
    run_id: str,
    config: ResolvedRunConfig,
    checkpoint_relative: str | None = None,
    completed_iterations: int = 0,
) -> None:
    path = run_dir / "run_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid run manifest: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("run manifest must be a JSON object")
    previous_run_id = payload.get("run_id")
    if previous_run_id is not None and previous_run_id != run_id:
        raise RuntimeError("run directory belongs to a different source run")
    payload["run_id"] = run_id
    payload["effective_training"] = {
        "maximum_iterations": config.run.maximum_iterations,
        "save_interval_iterations": config.run.save_interval_iterations,
        "num_envs": config.platform.num_envs,
        "worker_threads": config.platform.worker_threads,
        "rollout_steps_per_env": config.ppo.rollout_steps_per_env,
        "mini_batches": config.ppo.mini_batches,
    }
    payload["episode_boundary_semantics"] = EPISODE_BOUNDARY_SEMANTICS
    payload["completed_iterations"] = completed_iterations
    references = payload.get("checkpoint_references", [])
    if not isinstance(references, list):
        raise RuntimeError("run manifest checkpoint_references must be a list")
    if checkpoint_relative is not None and checkpoint_relative not in references:
        references.append(checkpoint_relative)
    payload["checkpoint_references"] = references
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _prepare_run(
    config: ResolvedRunConfig,
    *,
    resume: ValidatedCheckpoint | None,
    dependencies: Mapping[str, object],
) -> tuple[Path, str]:
    run_dir = Path(config.run.output_dir).resolve()
    if resume is None:
        prepare_run_directory(
            config,
            run_dir,
            dependencies=tuple(
                f"{name}={version}" for name, version in cast(Mapping[str, str], dependencies["packages"]).items()
            ),
        )
        run_id = uuid.uuid4().hex
        _update_run_manifest(run_dir, run_id=run_id, config=config)
        return run_dir, run_id
    source_run = resume.paths.payload.parent.parent.resolve()
    if run_dir != source_run:
        raise CheckpointError("resume output_dir must be the checkpoint's source run directory")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise CheckpointError("resume source run has no run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("run_id") != resume.sidecar["run_id"]:
        raise CheckpointError("resume checkpoint and run manifest have different run_id values")
    return run_dir, cast(str, resume.sidecar["run_id"])


def _save_iteration_checkpoint(
    run_dir: Path,
    run_id: str,
    config: ResolvedRunConfig,
    components: TrainingComponents,
    environment: TaskVecEnv,
    telemetry: TelemetryAggregator,
    state: RunnerState,
    *,
    kind: str,
    dependencies: Mapping[str, object],
) -> Path:
    storage = _storage(components.ppo)
    curriculum_state, environment_states = environment.checkpoint_rng_state()
    stem = checkpoint_name(state.completed_iterations, final=kind == "final")
    target = run_dir / "checkpoints" / f"{stem}.pt"
    sidecar = build_sidecar_metadata(
        config,
        iteration=state.completed_iterations,
        kind=kind,
        run_id=run_id,
        dependencies=dependencies,
        storage_metadata=storage.checkpoint_metadata(),
    )
    saved = save_checkpoint(
        target,
        sidecar_metadata=sidecar,
        model_state_dict=components.policy.state_dict(),
        optimizer_state_dict=components.ppo.optimizer.state_dict(),
        ppo_state_dict=components.ppo.get_checkpoint_state(),
        rollout_storage_metadata=storage.checkpoint_metadata(),
        policy_action_space=components.policy.action_space_metadata(),
        runner_state_dict=state.to_dict(),
        telemetry_state_dict=telemetry.state_dict(),
        process_rng_state_dict=capture_process_rng_state(),
        curriculum_state_dict=curriculum_state,
        environment_rng_state_dicts=environment_states,
        overwrite=kind == "final",
    )
    relative = saved.payload.relative_to(run_dir).as_posix()
    _update_run_manifest(
        run_dir,
        run_id=run_id,
        config=config,
        checkpoint_relative=relative,
        completed_iterations=state.completed_iterations,
    )
    if kind == "periodic":
        prune_periodic_checkpoints(saved.payload.parent, keep_last=5)
    return saved.payload


def _restore_training_state(
    validated: ValidatedCheckpoint,
    components: TrainingComponents,
    environment: TaskVecEnv,
    telemetry: TelemetryAggregator,
) -> RunnerState:
    payload = validated.payload
    storage = _storage(components.ppo)
    storage.validate_checkpoint_metadata(payload["rollout_storage_metadata"])
    if payload["policy_action_space"] != components.policy.action_space_metadata():
        raise CheckpointError("checkpoint policy action space differs from the seven-joint arm contract")
    model_state = payload["model_state_dict"]
    optimizer_state = payload["optimizer_state_dict"]
    ppo_state = payload["ppo_state_dict"]
    if (
        not isinstance(model_state, Mapping)
        or not isinstance(optimizer_state, Mapping)
        or not isinstance(ppo_state, Mapping)
    ):
        raise CheckpointError("checkpoint model/optimizer/PPO state is malformed")
    components.policy.load_state_dict(model_state, strict=True)
    components.ppo.optimizer.load_state_dict(dict(optimizer_state))
    components.ppo.load_checkpoint_state(dict(ppo_state))
    runner_raw = payload["runner_state_dict"]
    telemetry_raw = payload["telemetry_state_dict"]
    process_rng = payload["process_rng_state_dict"]
    curriculum = payload["curriculum_state_dict"]
    environment_states = payload["environment_rng_state_dicts"]
    if (
        not isinstance(runner_raw, Mapping)
        or not isinstance(telemetry_raw, Mapping)
        or not isinstance(process_rng, Mapping)
        or not isinstance(curriculum, Mapping)
    ):
        raise CheckpointError("checkpoint runner/RNG/curriculum state is malformed")
    if not isinstance(environment_states, Sequence):
        raise CheckpointError("checkpoint environment RNG state is malformed")
    state = RunnerState.from_dict(runner_raw)
    telemetry.load_state_dict(telemetry_raw)
    if state.completed_iterations != validated.iteration or components.ppo.update_count != validated.iteration:
        raise CheckpointError("checkpoint iteration, PPO update count, and runner state disagree")
    restore_process_rng_state(process_rng)
    environment.restore_checkpoint_rng_state(curriculum, environment_states)
    _finite_optimizer_state(components.ppo)
    return state


def _preflight_resume_components(validated: ValidatedCheckpoint, components: TrainingComponents) -> None:
    """Perform storage/action checks before TaskVecEnv is constructed."""

    storage = _storage(components.ppo)
    storage.validate_checkpoint_metadata(validated.payload["rollout_storage_metadata"])
    if validated.payload["policy_action_space"] != components.policy.action_space_metadata():
        raise CheckpointError("checkpoint is not the exact seven-joint arm policy contract")


def train(config: ResolvedRunConfig, *, resume_path: str | Path | None = None) -> RunnerState:
    """Run bounded/configured training; callers choose any short smoke overrides."""

    config.validate()
    dependencies = dependency_snapshot()
    validated = (
        None
        if resume_path is None
        else load_checkpoint_preflight(
            resume_path,
            config,
        )
    )
    if resume_path is None and config.run.checkpoint_source != "none":
        raise CheckpointError("training warm starts are forbidden; use --resume with a MuJoCo checkpoint")
    seed_all(config.run.seed)
    components = build_training_components(config)
    if validated is not None:
        _preflight_resume_components(validated, components)
    run_dir, run_id = _prepare_run(config, resume=validated, dependencies=dependencies)
    telemetry = TelemetryAggregator()
    metadata = ReportMetadata.from_config(config)
    process_start = time.perf_counter()
    cpu_start = time.process_time()
    with (
        TaskVecEnv(config) as environment,
        TensorBoardTelemetryWriter(run_dir / "tensorboard") as tensorboard,
    ):
        if validated is None:
            state = RunnerState()
            observations, _ = environment.reset()
        else:
            state = _restore_training_state(validated, components, environment, telemetry)
            observations, _ = environment.discard_incomplete_episodes_and_reset_all()
        start_iteration = state.completed_iterations
        if start_iteration >= config.run.maximum_iterations:
            raise CheckpointError("checkpoint already reached configured maximum_iterations")
        for iteration_index in range(start_iteration, config.run.maximum_iterations):
            sampling_start = time.perf_counter()
            observations = collect_rollout(
                environment,
                components.ppo,
                observations,
                iteration=iteration_index + 1,
                telemetry=telemetry,
            )
            completed_episodes = int(_storage(components.ppo).dones.sum().item())
            state.sampling_seconds += time.perf_counter() - sampling_start
            learning_start = time.perf_counter()
            losses = components.ppo.update()
            state.learning_seconds += time.perf_counter() - learning_start
            _finite_optimizer_state(components.ppo)
            actor_gradient_norm = gradient_diagnostics(components.policy)
            if not math.isfinite(actor_gradient_norm):
                raise FloatingPointError("PPO gradient diagnostics are non-finite")
            state.completed_iterations = iteration_index + 1
            step_count = config.platform.num_envs * config.ppo.rollout_steps_per_env
            state.total_policy_steps += step_count
            state.total_physics_steps += step_count * config.physics.decimation
            state.completed_episodes += completed_episodes
            state.last_losses = {name: float(value) for name, value in losses.items()}
            state.last_actor_gradient_norm = actor_gradient_norm
            state.total_seconds = state.sampling_seconds + state.learning_seconds
            telemetry.record_ppo_update(
                PPOUpdateEvent(
                    iteration=state.completed_iterations,
                    policy_loss=float(losses["surrogate"]),
                    value_loss=float(losses["value_function"]),
                    entropy=float(losses["entropy"]),
                    kl=float(losses["kl"]),
                    total_loss=float(losses["surrogate"] + losses["value_function"]),
                    gradient_norm=actor_gradient_norm,
                    gradients_finite=True,
                    log_probability=float(losses["log_probability"]),
                )
            )
            elapsed = max(time.perf_counter() - process_start, torch.finfo(torch.float64).eps)
            completed_this_process = state.completed_iterations - start_iteration
            remaining = config.run.maximum_iterations - state.completed_iterations
            eta = remaining * elapsed / max(completed_this_process, 1)
            telemetry.record_run(
                RunEvent(
                    iteration=state.completed_iterations,
                    total_steps=state.total_policy_steps,
                    steps_per_second=state.total_policy_steps / max(state.total_seconds, 1.0e-12),
                    sampling_seconds=state.sampling_seconds,
                    learning_seconds=state.learning_seconds,
                    total_seconds=state.total_seconds,
                    eta_seconds=eta,
                    memory_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    cpu_percent=max(0.0, 100.0 * (time.process_time() - cpu_start) / elapsed),
                    num_envs=config.platform.num_envs,
                    worker_threads=config.platform.worker_threads,
                    physics_steps=state.total_physics_steps,
                    policy_steps=state.total_policy_steps,
                    sim_steps_per_second=state.total_physics_steps / max(state.total_seconds, 1.0e-12),
                    realtime_factor=(
                        state.total_physics_steps * config.physics.physics_dt_s / max(state.total_seconds, 1.0e-12)
                    ),
                    fixed_torsional_friction=config.randomization.fixed_torsional_friction,
                    fixed_rolling_friction=config.randomization.fixed_rolling_friction,
                    randomization_version=config.randomization.version,
                    bottleneck=("sampling" if state.sampling_seconds >= state.learning_seconds else "learning"),
                )
            )
            snapshot = telemetry.snapshot(state.completed_iterations)
            tensorboard.write_iteration(snapshot, metadata)
            write_json_report(
                run_dir / "metrics" / f"iteration_{state.completed_iterations:06d}.json",
                snapshot,
                metadata,
            )
            print(render_terminal_summary(snapshot).text, flush=True)
            periodic = (
                state.completed_iterations % config.run.save_interval_iterations == 0
                and state.completed_iterations < config.run.maximum_iterations
            )
            if periodic:
                _save_iteration_checkpoint(
                    run_dir,
                    run_id,
                    config,
                    components,
                    environment,
                    telemetry,
                    state,
                    kind="periodic",
                    dependencies=dependencies,
                )
                # Both uninterrupted and resumed branches consume exactly this reset.
                observations, _ = environment.discard_incomplete_episodes_and_reset_all()
        _save_iteration_checkpoint(
            run_dir,
            run_id,
            config,
            components,
            environment,
            telemetry,
            state,
            kind="final",
            dependencies=dependencies,
        )
        final_snapshot = telemetry.snapshot(state.completed_iterations)
        write_final_reports(run_dir, final_snapshot, metadata)
    finalize_run_manifest(run_dir)
    return state


def run(config: ResolvedRunConfig, args: argparse.Namespace) -> int:
    """Public ``python -m tienkung_rl.train`` runtime integration."""

    resolved = effective_config(config, args)
    resume_requested = bool(getattr(args, "resume", False))
    if resume_requested:
        checkpoint_path = resolved.evaluation.checkpoint_path
        if not checkpoint_path:
            raise CheckpointError("--resume requires evaluation.checkpoint_path to select a MuJoCo checkpoint")
        if resolved.run.checkpoint_source != "mujoco":
            raise CheckpointError("--resume requires run.checkpoint_source=mujoco")
        train(resolved, resume_path=checkpoint_path)
    else:
        train(resolved)
    return 0


__all__ = [
    "RNG_STATE_VERSION",
    "RUNNER_STATE_VERSION",
    "RunnerState",
    "TrainingComponents",
    "build_training_components",
    "capture_process_rng_state",
    "collect_rollout",
    "effective_config",
    "gradient_diagnostics",
    "restore_process_rng_state",
    "run",
    "seed_all",
    "terminal_bootstrap_values",
    "train",
]
