"""Save and load self-contained training checkpoints with atomic file replacement."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import torch

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import canonical_json

CHECKPOINT_SCHEMA_VERSION: Final = "tienkung_training_checkpoint_v3"
CHECKPOINT_SIDECAR_VERSION: Final = "tienkung_training_checkpoint_sidecar_v3"
EPISODE_BOUNDARY_SEMANTICS: Final = "discard_incomplete_and_reset_all"
RETENTION_POLICY: Final = "keep_last_5_periodic_plus_final_v1"


class CheckpointError(RuntimeError):
    """A checkpoint is incomplete, corrupt, incompatible, or unsafe."""


@dataclass(frozen=True, slots=True)
class CheckpointPaths:
    payload: Path
    sidecar: Path


@dataclass(frozen=True, slots=True)
class ValidatedCheckpoint:
    """A loaded payload and its training metadata."""

    paths: CheckpointPaths
    sidecar: Mapping[str, object]
    payload: Mapping[str, object]

    @property
    def iteration(self) -> int:
        return cast(int, self.payload["iteration"])


def checkpoint_paths(path: str | Path) -> CheckpointPaths:
    candidate = Path(path)
    if candidate.suffix == ".json":
        candidate = candidate.with_suffix(".pt")
    elif candidate.suffix != ".pt":
        raise CheckpointError("checkpoint path must end in .pt or .json")
    return CheckpointPaths(candidate, candidate.with_suffix(".json"))


def checkpoint_name(iteration: int, *, final: bool = False) -> str:
    if type(iteration) is not int or iteration < 1:
        raise ValueError("checkpoint iteration must be a positive integer")
    return "model_final" if final else f"model_{iteration:06d}"


def dependency_snapshot() -> dict[str, object]:
    """Return the exact CPU runtime identity persisted in every sidecar."""

    packages: dict[str, str] = {}
    for distribution in ("mujoco", "numpy", "torch", "PyYAML", "tensorboard"):
        try:
            packages[distribution.casefold()] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution.casefold()] = "missing"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
    }


def build_sidecar_metadata(
    config: ResolvedRunConfig,
    *,
    iteration: int,
    kind: str,
    run_id: str,
    dependencies: Mapping[str, object],
    storage_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build the immutable, JSON-readable half of a checkpoint bundle."""

    config.validate()
    if kind not in {"periodic", "final"}:
        raise ValueError("checkpoint kind must be periodic or final")
    if type(iteration) is not int or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    base: dict[str, object] = {
        "schema_version": CHECKPOINT_SIDECAR_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_simulator": "mujoco",
        "checkpoint_kind": kind,
        "iteration": iteration,
        "run_id": run_id,
        "command": config.run.command,
        "observation_dimension": config.task.observation.dimension,
        "action_dimension": config.task.action.dimension,
        "option_version": config.task.option.version,
        "dependency_snapshot": dict(dependencies),
        "storage_metadata": dict(storage_metadata),
        "randomization": {
            "version": config.randomization.version,
            "five_independent_streams_per_environment": True,
            "executor_pcg64_stream_per_environment": True,
            "fixed_torsional_friction": config.randomization.fixed_torsional_friction,
            "fixed_rolling_friction": config.randomization.fixed_rolling_friction,
        },
        "training": {
            "seed": config.run.seed,
            "num_envs": config.platform.num_envs,
            "worker_threads": config.platform.worker_threads,
            "rollout_steps_per_env": config.ppo.rollout_steps_per_env,
            "learning_epochs": config.ppo.learning_epochs,
            "mini_batches": config.ppo.mini_batches,
            "maximum_iterations": config.run.maximum_iterations,
            "save_interval_iterations": config.run.save_interval_iterations,
            "actor_learning_rate": config.ppo.actor_learning_rate,
            "critic_learning_rate": config.ppo.critic_learning_rate,
            "device": "cpu",
        },
        "episode_boundary_semantics": EPISODE_BOUNDARY_SEMANTICS,
        "resume_requires_uniform_reset": True,
        "retention_policy": RETENTION_POLICY,
    }
    return base


_PAYLOAD_FIELDS: Final = {
    "schema_version",
    "source_simulator",
    "checkpoint_kind",
    "iteration",
    "run_id",
    "episode_boundary_semantics",
    "model_state_dict",
    "optimizer_state_dict",
    "ppo_state_dict",
    "rollout_storage_metadata",
    "policy_action_space",
    "runner_state_dict",
    "telemetry_state_dict",
    "process_rng_state_dict",
    "curriculum_state_dict",
    "environment_rng_state_dicts",
}


def _validate_payload_structure(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload root must be a mapping")
    if not _PAYLOAD_FIELDS.issubset(payload):
        missing = sorted(_PAYLOAD_FIELDS - set(payload))
        raise CheckpointError(f"checkpoint payload is missing fields: {missing}")
    if payload["schema_version"] not in {CHECKPOINT_SCHEMA_VERSION, "tienkung_training_checkpoint_v2"}:
        raise CheckpointError("unknown checkpoint payload schema")
    if payload["source_simulator"] != "mujoco":
        source = payload["source_simulator"]
        if isinstance(source, str) and ("isaac" in source.casefold() or "physx" in source.casefold()):
            raise CheckpointError("Isaac/PhysX checkpoints are forbidden")
        raise CheckpointError("unknown checkpoint source simulator")
    if payload["episode_boundary_semantics"] != EPISODE_BOUNDARY_SEMANTICS:
        raise CheckpointError("checkpoint has incompatible episode boundary semantics")
    if type(payload["iteration"]) is not int or payload["iteration"] < 1:
        raise CheckpointError("checkpoint iteration must be a positive integer")
    for name in ("run_id",):
        value = payload[name]
        if not isinstance(value, str) or not value:
            raise CheckpointError(f"checkpoint {name} must be a non-empty string")
    for name in (
        "model_state_dict",
        "optimizer_state_dict",
        "ppo_state_dict",
        "rollout_storage_metadata",
        "policy_action_space",
        "runner_state_dict",
        "telemetry_state_dict",
        "process_rng_state_dict",
        "curriculum_state_dict",
    ):
        if not isinstance(payload[name], Mapping):
            raise CheckpointError(f"checkpoint {name} must be a mapping")
    states = payload["environment_rng_state_dicts"]
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise CheckpointError("checkpoint environment RNG states must be a sequence")
    return cast(Mapping[str, object], payload)


def save_checkpoint(
    path: str | Path,
    *,
    sidecar_metadata: Mapping[str, object],
    model_state_dict: Mapping[str, object],
    optimizer_state_dict: Mapping[str, object],
    ppo_state_dict: Mapping[str, object],
    rollout_storage_metadata: Mapping[str, object],
    policy_action_space: Mapping[str, object],
    runner_state_dict: Mapping[str, object],
    telemetry_state_dict: Mapping[str, object],
    process_rng_state_dict: Mapping[str, object],
    curriculum_state_dict: Mapping[str, object],
    environment_rng_state_dicts: Sequence[object],
    overwrite: bool = False,
) -> CheckpointPaths:
    """Write training state atomically, with an optional readable JSON export."""

    paths = checkpoint_paths(path)
    paths.payload.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (paths.payload.exists() or paths.sidecar.exists()):
        raise FileExistsError(f"refusing to overwrite checkpoint bundle {paths.payload.stem}")
    required_sidecar = {
        "schema_version",
        "source_simulator",
        "checkpoint_kind",
        "iteration",
        "run_id",
        "episode_boundary_semantics",
    }
    if not required_sidecar.issubset(sidecar_metadata):
        raise CheckpointError("checkpoint sidecar metadata is incomplete")
    sidecar_core = dict(sidecar_metadata)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_simulator": "mujoco",
        "checkpoint_kind": sidecar_core["checkpoint_kind"],
        "iteration": sidecar_core["iteration"],
        "run_id": sidecar_core["run_id"],
        "episode_boundary_semantics": EPISODE_BOUNDARY_SEMANTICS,
        "model_state_dict": dict(model_state_dict),
        "optimizer_state_dict": dict(optimizer_state_dict),
        "ppo_state_dict": dict(ppo_state_dict),
        "rollout_storage_metadata": dict(rollout_storage_metadata),
        "policy_action_space": dict(policy_action_space),
        "runner_state_dict": dict(runner_state_dict),
        "telemetry_state_dict": dict(telemetry_state_dict),
        "process_rng_state_dict": dict(process_rng_state_dict),
        "curriculum_state_dict": dict(curriculum_state_dict),
        "environment_rng_state_dicts": tuple(environment_rng_state_dicts),
        "metadata": sidecar_core,
    }
    _validate_payload_structure(payload)
    token = uuid.uuid4().hex
    payload_temp = paths.payload.with_name(f".{paths.payload.name}.{token}.tmp")
    sidecar_temp = paths.sidecar.with_name(f".{paths.sidecar.name}.{token}.tmp")
    try:
        torch.save(payload, payload_temp)
        with payload_temp.open("rb") as stream:
            os.fsync(stream.fileno())
        sidecar_bytes = (canonical_json(sidecar_core) + "\n").encode("utf-8")
        with sidecar_temp.open("xb") as stream:
            stream.write(sidecar_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(payload_temp, paths.payload)
        os.replace(sidecar_temp, paths.sidecar)
        directory_fd = os.open(paths.payload.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for temporary in (payload_temp, sidecar_temp):
            with suppress(FileNotFoundError):
                temporary.unlink()
    return paths


def load_checkpoint_preflight(
    path: str | Path,
    config: ResolvedRunConfig,
    *,
    mode: str = "resume",
) -> ValidatedCheckpoint:
    """Load weights and training state; JSON is an optional readable export.

    Old payloads can obtain metadata from their original JSON sidecar.
    Tensor dimensions and action semantics are checked by the consumer.
    """
    if mode not in {"resume", "inference"}:
        raise ValueError("checkpoint load mode must be 'resume' or 'inference'")
    paths = checkpoint_paths(path)
    try:
        loaded = torch.load(paths.payload, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CheckpointError(f"cannot deserialize checkpoint {paths.payload}: {error}") from error
    payload = _validate_payload_structure(loaded)
    metadata = payload.get("metadata")
    if metadata is None and paths.sidecar.is_file():
        metadata = json.loads(paths.sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        # Older checkpoints without a sidecar still have the state needed by the runner.
        metadata = {
            "run_id": payload["run_id"],
            "storage_metadata": payload["rollout_storage_metadata"],
        }
    return ValidatedCheckpoint(paths, metadata, payload)


def prune_periodic_checkpoints(directory: str | Path, *, keep_last: int = 5) -> tuple[str, ...]:
    """Remove old periodic checkpoints while preserving the final checkpoint."""

    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be a positive integer")
    target = Path(directory)
    pairs: list[tuple[int, CheckpointPaths]] = []
    for payload in target.glob("model_[0-9][0-9][0-9][0-9][0-9][0-9].pt"):
        paths = checkpoint_paths(payload)
        try:
            iteration = int(payload.stem.removeprefix("model_"))
        except ValueError:
            continue
        pairs.append((iteration, paths))
    removed: list[str] = []
    for _, paths in sorted(pairs)[:-keep_last]:
        paths.payload.unlink()
        paths.sidecar.unlink(missing_ok=True)
        removed.append(paths.payload.stem)
    return tuple(removed)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_SIDECAR_VERSION",
    "EPISODE_BOUNDARY_SEMANTICS",
    "RETENTION_POLICY",
    "CheckpointError",
    "CheckpointPaths",
    "ValidatedCheckpoint",
    "build_sidecar_metadata",
    "checkpoint_name",
    "checkpoint_paths",
    "dependency_snapshot",
    "load_checkpoint_preflight",
    "prune_periodic_checkpoints",
    "save_checkpoint",
]
