"""Deterministic in-process parallel execution for native MuJoCo environments.

The executor deliberately remains synchronous at its public boundary.  Work is
submitted to a persistent thread pool, every active worker reaches a barrier,
and only then are immutable per-environment results copied into contiguous
batch buffers in increasing ``env_id`` order.  Each environment owns its own
backend and NumPy generator; thread scheduling therefore cannot choose either
the return order or another environment's random stream.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import json
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import sha256_fingerprint

from .backend import EnvironmentBackend
from .mujoco_backend import MujocoEnvironmentBackend
from .state import (
    ActuatorControlRanges,
    ContactFrame,
    PhysicsState,
    PhysicsSubstepTrace,
    ResetSpec,
    SimulationDiagnostics,
    SiteHingeKinematics,
    readonly_float64,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BackendFactory = Callable[[int], EnvironmentBackend]


class _EnvironmentIdentified(Protocol):
    @property
    def env_id(self) -> int: ...


_ResultT = TypeVar("_ResultT", bound=_EnvironmentIdentified)


class ResetSpecFactory(Protocol):
    """Build one reset from the environment-owned RNG.

    Implementations must be stateless.  Randomness must be drawn exclusively
    from ``rng`` so replay depends on the environment lifecycle, not the worker
    thread that happens to execute the callback.
    """

    def __call__(self, env_id: int, rng: np.random.Generator) -> ResetSpec: ...


class ParallelExecutorError(RuntimeError):
    """Base class for parallel lifecycle and worker failures."""


class ParallelExecutorClosedError(ParallelExecutorError):
    """Work was requested after shutdown began."""


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Stable main-thread representation of one worker exception."""

    env_id: int
    physics_step: int
    exception_type: str
    message: str
    config_fingerprint: str

    def __post_init__(self) -> None:
        if self.env_id < 0 or self.physics_step < 0:
            raise ValueError("worker failure env_id and physics_step must be non-negative")
        if not self.exception_type or not self.message or not self.config_fingerprint:
            raise ValueError("worker failure type, message and fingerprint must not be empty")


class ParallelWorkerError(ParallelExecutorError):
    """One or more workers failed after every submitted future was collected."""

    def __init__(self, failures: Sequence[WorkerFailure]) -> None:
        ordered = tuple(sorted(failures, key=lambda item: item.env_id))
        if not ordered:
            raise ValueError("ParallelWorkerError requires at least one failure")
        self.failures = ordered
        first = ordered[0]
        summary = "; ".join(
            f"env_id={item.env_id} physics_step={item.physics_step} "
            f"{item.exception_type}: {item.message} fingerprint={item.config_fingerprint}"
            for item in ordered
        )
        super().__init__(f"{len(ordered)} parallel worker(s) failed; first env_id={first.env_id}: {summary}")


@dataclass(frozen=True, slots=True)
class ParallelResetRequest:
    """Reset one environment, optionally with an explicit deterministic spec."""

    env_id: int
    reset_spec: ResetSpec | None = None

    def __post_init__(self) -> None:
        if self.env_id < 0:
            raise ValueError("reset request env_id must be non-negative")
        if self.reset_spec is not None and not isinstance(self.reset_spec, ResetSpec):
            raise TypeError("reset_spec must be ResetSpec or None")


@dataclass(frozen=True, slots=True)
class ParallelStepRequest:
    """Apply one 13-dimensional target and advance one policy step."""

    env_id: int
    targets_rad: FloatArray
    substeps: int = 4

    def __post_init__(self) -> None:
        if self.env_id < 0:
            raise ValueError("step request env_id must be non-negative")
        if type(self.substeps) is not int or self.substeps < 1:
            raise ValueError("step request substeps must be a positive integer")
        object.__setattr__(
            self,
            "targets_rad",
            readonly_float64(self.targets_rad, (13,), "targets_rad"),
        )


@dataclass(frozen=True, slots=True)
class ParallelResetResult:
    """One reset result with explicit ordering and diagnostic context."""

    env_id: int
    state: PhysicsState
    diagnostics: SimulationDiagnostics
    physics_step: int


@dataclass(frozen=True, slots=True)
class ParallelStepResult:
    """One policy-step result with aggregated contacts and physical step count."""

    env_id: int
    state: PhysicsState
    contacts: ContactFrame
    diagnostics: SimulationDiagnostics
    substeps_executed: int
    physics_step: int
    substep_trace: PhysicsSubstepTrace


@dataclass(frozen=True, slots=True)
class ParallelActuatorControlRanges:
    """One environment's immutable named actuator bounds."""

    env_id: int
    ranges: ActuatorControlRanges


@dataclass(frozen=True, slots=True)
class ParallelSiteHingeKinematics:
    """One environment's immutable named site-hinge query result."""

    env_id: int
    kinematics: SiteHingeKinematics


@dataclass(frozen=True, slots=True)
class ParallelArmSiteJacobian:
    """One environment's immutable arm position and site-Jacobian query."""

    env_id: int
    joint_positions_rad: FloatArray
    site_jacobian_6x7: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_positions_rad",
            readonly_float64(self.joint_positions_rad, (7,), "joint_positions_rad"),
        )
        object.__setattr__(
            self,
            "site_jacobian_6x7",
            readonly_float64(self.site_jacobian_6x7, (6, 7), "site_jacobian_6x7"),
        )


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class BatchedPhysicsState:
    """Read-only views over one stable, contiguous batch-buffer lease.

    A lease is never recycled while this object is alive.  Long-lived callers
    may retain old batches safely; the executor allocates another storage only
    when both steady-state buffers are still leased.
    """

    generation: int
    env_ids: IntArray
    joint_names: tuple[str, ...]
    joint_positions_rad: FloatArray
    joint_velocities_rad_s: FloatArray
    body_names: tuple[str, ...]
    body_positions_m: FloatArray
    body_quaternions_wxyz: FloatArray
    site_names: tuple[str, ...]
    site_positions_m: FloatArray
    site_quaternions_wxyz: FloatArray
    site_linear_velocities_m_s: FloatArray
    site_angular_velocities_rad_s: FloatArray
    tcp_position_m: FloatArray
    tcp_quaternion_wxyz: FloatArray
    grasp_center_position_m: FloatArray
    grasp_center_quaternion_wxyz: FloatArray
    cylinder_position_m: FloatArray
    cylinder_quaternion_wxyz: FloatArray
    cylinder_linear_velocity_m_s: FloatArray
    cylinder_angular_velocity_rad_s: FloatArray
    actuator_names: tuple[str, ...]
    actuator_targets_rad: FloatArray
    actuator_forces_n_m: FloatArray
    cylinder_parameters: FloatArray
    episode_physics_steps: IntArray
    total_physics_steps: IntArray
    simulation_time_s: FloatArray
    total_simulation_time_s: FloatArray


@dataclass(frozen=True, slots=True)
class BatchResetResult:
    """Barrier-complete reset results sorted by environment ID."""

    results: tuple[ParallelResetResult, ...]
    state: BatchedPhysicsState


@dataclass(frozen=True, slots=True)
class BatchStepResult:
    """Barrier-complete step results sorted by environment ID."""

    results: tuple[ParallelStepResult, ...]
    state: BatchedPhysicsState


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    """Opaque identities used to audit absence of mutable sharing."""

    env_id: int
    backend_identity: int
    model_identity: int
    data_identity: int
    rng_identity: int


@dataclass(frozen=True, slots=True)
class EnvironmentRngManifest:
    """JSON-safe checkpoint/replay record for one environment RNG."""

    env_id: int
    root_seed: int
    bit_generator: str
    state_json: str
    successful_resets: int

    def __post_init__(self) -> None:
        if self.env_id < 0 or self.root_seed < 0 or self.successful_resets < 0:
            raise ValueError("RNG manifest counters must be non-negative")
        if not self.bit_generator or not self.state_json:
            raise ValueError("RNG manifest generator and state must not be empty")


@dataclass(slots=True)
class _EnvironmentSlot:
    env_id: int
    backend: EnvironmentBackend
    rng: np.random.Generator
    successful_resets: int = 0
    last_physics_step: int = 0


class _WorkerTaskError(Exception):
    def __init__(self, failure: WorkerFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


_FLOAT_STATE_FIELDS = (
    "joint_positions_rad",
    "joint_velocities_rad_s",
    "body_positions_m",
    "body_quaternions_wxyz",
    "site_positions_m",
    "site_quaternions_wxyz",
    "site_linear_velocities_m_s",
    "site_angular_velocities_rad_s",
    "tcp_position_m",
    "tcp_quaternion_wxyz",
    "grasp_center_position_m",
    "grasp_center_quaternion_wxyz",
    "cylinder_position_m",
    "cylinder_quaternion_wxyz",
    "cylinder_linear_velocity_m_s",
    "cylinder_angular_velocity_rad_s",
    "actuator_targets_rad",
    "actuator_forces_n_m",
)
_CYLINDER_PARAMETER_FIELDS = (
    "radius_m",
    "height_m",
    "mass_kg",
    "hand_sliding_friction",
    "table_sliding_friction",
    "torsional_friction",
    "rolling_friction",
)


class _BatchStateStorage:
    """Mutable storage that is only filled while no published lease exists."""

    def __init__(self, capacity: int, template: PhysicsState) -> None:
        self.capacity = capacity
        self.joint_names = template.joint_names
        self.body_names = template.body_names
        self.site_names = template.site_names
        self.actuator_names = template.actuator_names
        self.env_ids = np.empty(capacity, dtype=np.int64)
        self.float_arrays: dict[str, FloatArray] = {}
        for name in _FLOAT_STATE_FIELDS:
            source = cast(FloatArray, getattr(template, name))
            self.float_arrays[name] = np.empty((capacity, *source.shape), dtype=np.float64, order="C")
        self.cylinder_parameters = np.empty((capacity, len(_CYLINDER_PARAMETER_FIELDS)), dtype=np.float64, order="C")
        self.episode_physics_steps = np.empty(capacity, dtype=np.int64)
        self.total_physics_steps = np.empty(capacity, dtype=np.int64)
        self.simulation_time_s = np.empty(capacity, dtype=np.float64)
        self.total_simulation_time_s = np.empty(capacity, dtype=np.float64)
        self.lease: weakref.ReferenceType[BatchedPhysicsState] | None = None

    @property
    def available(self) -> bool:
        return self.lease is None or self.lease() is None

    def fill(
        self,
        generation: int,
        env_ids: Sequence[int],
        states: Sequence[PhysicsState],
    ) -> BatchedPhysicsState:
        count = len(states)
        if count != self.capacity or len(env_ids) != count:
            raise ValueError("batch buffer capacity differs from result count")
        if not self.available:
            raise RuntimeError("attempted to write a leased batch buffer")
        arrays: tuple[NDArray[np.generic], ...] = (
            self.env_ids,
            *self.float_arrays.values(),
            self.cylinder_parameters,
            self.episode_physics_steps,
            self.total_physics_steps,
            self.simulation_time_s,
            self.total_simulation_time_s,
        )
        for array in arrays:
            array.setflags(write=True)
        self.env_ids[:] = env_ids
        for row, state in enumerate(states):
            if (
                state.joint_names != self.joint_names
                or state.body_names != self.body_names
                or state.site_names != self.site_names
                or state.actuator_names != self.actuator_names
            ):
                raise ValueError("environment state name/order contract changed within a batch")
            for name, target in self.float_arrays.items():
                np.copyto(target[row], cast(FloatArray, getattr(state, name)))
            parameters = state.cylinder_parameters
            self.cylinder_parameters[row] = tuple(
                float(getattr(parameters, name)) for name in _CYLINDER_PARAMETER_FIELDS
            )
            self.episode_physics_steps[row] = state.episode_physics_steps
            self.total_physics_steps[row] = state.total_physics_steps
            self.simulation_time_s[row] = state.simulation_time_s
            self.total_simulation_time_s[row] = state.total_simulation_time_s
        for array in arrays:
            if not array.flags.c_contiguous:
                raise RuntimeError("internal batch state buffer lost C-contiguous layout")
            array.setflags(write=False)
        batch = BatchedPhysicsState(
            generation=generation,
            env_ids=self.env_ids,
            joint_names=self.joint_names,
            joint_positions_rad=self.float_arrays["joint_positions_rad"],
            joint_velocities_rad_s=self.float_arrays["joint_velocities_rad_s"],
            body_names=self.body_names,
            body_positions_m=self.float_arrays["body_positions_m"],
            body_quaternions_wxyz=self.float_arrays["body_quaternions_wxyz"],
            site_names=self.site_names,
            site_positions_m=self.float_arrays["site_positions_m"],
            site_quaternions_wxyz=self.float_arrays["site_quaternions_wxyz"],
            site_linear_velocities_m_s=self.float_arrays["site_linear_velocities_m_s"],
            site_angular_velocities_rad_s=self.float_arrays["site_angular_velocities_rad_s"],
            tcp_position_m=self.float_arrays["tcp_position_m"],
            tcp_quaternion_wxyz=self.float_arrays["tcp_quaternion_wxyz"],
            grasp_center_position_m=self.float_arrays["grasp_center_position_m"],
            grasp_center_quaternion_wxyz=self.float_arrays["grasp_center_quaternion_wxyz"],
            cylinder_position_m=self.float_arrays["cylinder_position_m"],
            cylinder_quaternion_wxyz=self.float_arrays["cylinder_quaternion_wxyz"],
            cylinder_linear_velocity_m_s=self.float_arrays["cylinder_linear_velocity_m_s"],
            cylinder_angular_velocity_rad_s=self.float_arrays["cylinder_angular_velocity_rad_s"],
            actuator_names=self.actuator_names,
            actuator_targets_rad=self.float_arrays["actuator_targets_rad"],
            actuator_forces_n_m=self.float_arrays["actuator_forces_n_m"],
            cylinder_parameters=self.cylinder_parameters,
            episode_physics_steps=self.episode_physics_steps,
            total_physics_steps=self.total_physics_steps,
            simulation_time_s=self.simulation_time_s,
            total_simulation_time_s=self.total_simulation_time_s,
        )
        self.lease = weakref.ref(batch)
        return batch


class _BatchBufferPool:
    def __init__(self, template: PhysicsState) -> None:
        self._template = template
        self._by_capacity: dict[int, list[_BatchStateStorage]] = {}
        self._allocations = 0

    @property
    def allocations(self) -> int:
        return self._allocations

    def publish(
        self,
        generation: int,
        env_ids: Sequence[int],
        states: Sequence[PhysicsState],
    ) -> BatchedPhysicsState:
        capacity = len(states)
        storages = self._by_capacity.setdefault(capacity, [])
        storage = next((candidate for candidate in storages if candidate.available), None)
        if storage is None:
            storage = _BatchStateStorage(capacity, self._template)
            storages.append(storage)
            self._allocations += 1
        return storage.fill(generation, env_ids, states)


_EXECUTOR_IDS = itertools.count()


class ThreadPoolParallelExecutor:
    """Persistent deterministic executor for isolated MuJoCo environments."""

    def __init__(
        self,
        config: ResolvedRunConfig,
        *,
        num_envs: int | None = None,
        worker_threads: int | None = None,
        seed: int | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        config.validate()
        self._num_envs = config.platform.num_envs if num_envs is None else num_envs
        self._worker_threads = config.platform.worker_threads if worker_threads is None else worker_threads
        self._root_seed = config.run.seed if seed is None else seed
        if type(self._num_envs) is not int or self._num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(self._worker_threads) is not int or not 1 <= self._worker_threads <= self._num_envs:
            raise ValueError("worker_threads must lie in [1, num_envs]")
        if type(self._root_seed) is not int or not 0 <= self._root_seed <= 2**63 - 1:
            raise ValueError("seed must lie in [0, 2^63-1]")
        self._config_fingerprint = sha256_fingerprint(config)
        self._operation_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._generation = 0
        executor_id = next(_EXECUTOR_IDS)
        self._thread_name_prefix = f"tienkung-mujoco-{executor_id}"
        self._pool = ThreadPoolExecutor(
            max_workers=self._worker_threads,
            thread_name_prefix=self._thread_name_prefix,
        )

        factory: BackendFactory
        if backend_factory is None:

            def default_factory(env_id: int) -> EnvironmentBackend:
                del env_id
                return MujocoEnvironmentBackend(
                    config,
                    config_fingerprint=self._config_fingerprint,
                )

            factory = default_factory
        else:
            factory = backend_factory

        self._slots: tuple[_EnvironmentSlot, ...] = ()
        created: list[EnvironmentBackend] = []
        try:
            slots: list[_EnvironmentSlot] = []
            failures: list[WorkerFailure] = []
            # MuJoCo model loading and PyYAML's C loader both touch native
            # process-global state.  Construct isolated backends serially;
            # only lifecycle operations run in the persistent worker pool.
            for env_id in range(self._num_envs):
                try:
                    backend = factory(env_id)
                    if not isinstance(backend, EnvironmentBackend):
                        raise TypeError("backend_factory returned an object outside EnvironmentBackend")
                    created.append(backend)
                    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence((self._root_seed, env_id))))
                    slots.append(_EnvironmentSlot(env_id=env_id, backend=backend, rng=generator))
                except Exception as error:
                    failures.append(self._plain_failure(env_id, 0, error))
            if failures:
                raise ParallelWorkerError(failures)
            self._slots = tuple(sorted(slots, key=lambda slot: slot.env_id))
            backend_identities = tuple(id(slot.backend) for slot in self._slots)
            if len(set(backend_identities)) != self._num_envs:
                raise TypeError("backend_factory must return one independent backend per env_id")
            model_identities = tuple(
                id(model) for slot in self._slots if (model := getattr(slot.backend, "_model", None)) is not None
            )
            data_identities = tuple(
                id(data) for slot in self._slots if (data := getattr(slot.backend, "_data", None)) is not None
            )
            if len(set(model_identities)) != len(model_identities):
                raise TypeError("backend_factory returned environments sharing mutable MjModel data")
            if len(set(data_identities)) != len(data_identities):
                raise TypeError("backend_factory returned environments sharing mutable MjData state")
            template = self._slots[0].backend.read_state()
            self._buffers = _BatchBufferPool(template)
        except Exception:
            for backend in created:
                with contextlib.suppress(Exception):
                    backend.close()
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._closed = True
            raise

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def worker_threads(self) -> int:
        return self._worker_threads

    @property
    def thread_name_prefix(self) -> str:
        return self._thread_name_prefix

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    @property
    def is_closed(self) -> bool:
        with self._state_lock:
            return self._closed

    @property
    def buffer_allocation_count(self) -> int:
        return self._buffers.allocations

    @property
    def environment_identities(self) -> tuple[EnvironmentIdentity, ...]:
        with self._operation_lock:
            self._require_open()
            identities: list[EnvironmentIdentity] = []
            for slot in self._slots:
                model = getattr(slot.backend, "_model", None)
                data = getattr(slot.backend, "_data", None)
                identities.append(
                    EnvironmentIdentity(
                        env_id=slot.env_id,
                        backend_identity=id(slot.backend),
                        model_identity=id(model),
                        data_identity=id(data),
                        rng_identity=id(slot.rng),
                    )
                )
            return tuple(identities)

    def _require_open(self) -> None:
        with self._state_lock:
            if self._closing or self._closed:
                raise ParallelExecutorClosedError("parallel executor is closing or closed")

    def _validate_requests(self, requests: Iterable[_ResultT]) -> tuple[_ResultT, ...]:
        ordered = tuple(sorted(requests, key=lambda request: request.env_id))
        if not ordered:
            raise ValueError("parallel batch must contain at least one request")
        env_ids = tuple(request.env_id for request in ordered)
        if len(set(env_ids)) != len(env_ids):
            raise ValueError(f"parallel batch contains duplicate env_ids: {env_ids}")
        invalid = tuple(env_id for env_id in env_ids if not 0 <= env_id < self._num_envs)
        if invalid:
            raise ValueError(f"parallel batch contains out-of-range env_ids: {invalid}")
        return ordered

    def _plain_failure(self, env_id: int, physics_step: int, error: Exception) -> WorkerFailure:
        message = str(error).strip() or repr(error)
        return WorkerFailure(
            env_id=env_id,
            physics_step=max(0, physics_step),
            exception_type=type(error).__name__,
            message=message,
            config_fingerprint=self._config_fingerprint,
        )

    def _failure(self, slot: _EnvironmentSlot, error: Exception) -> WorkerFailure:
        diagnostics = getattr(error, "diagnostics", None)
        if isinstance(diagnostics, SimulationDiagnostics):
            physics_step = diagnostics.physics_step
            fingerprint = diagnostics.config_fingerprint
        else:
            physics_step = slot.last_physics_step
            fingerprint = self._config_fingerprint
            try:
                current = slot.backend.read_warnings()
                physics_step = current.physics_step
                fingerprint = current.config_fingerprint
            except Exception:
                pass
        message = str(error).strip() or repr(error)
        return WorkerFailure(
            env_id=slot.env_id,
            physics_step=max(0, physics_step),
            exception_type=type(error).__name__,
            message=message,
            config_fingerprint=fingerprint,
        )

    def _run_reset(
        self,
        request: ParallelResetRequest,
        reset_spec_factory: ResetSpecFactory | None,
    ) -> ParallelResetResult:
        slot = self._slots[request.env_id]
        saved_rng_state: Mapping[str, object] | None = None
        try:
            if request.reset_spec is not None:
                reset_spec = request.reset_spec
            elif reset_spec_factory is None:
                reset_spec = ResetSpec()
            else:
                saved_rng_state = copy.deepcopy(slot.rng.bit_generator.state)
                reset_spec = reset_spec_factory(slot.env_id, slot.rng)
                if not isinstance(reset_spec, ResetSpec):
                    raise TypeError("reset_spec_factory must return ResetSpec")
            state = slot.backend.reset(reset_spec)
            diagnostics = slot.backend.read_warnings()
            slot.successful_resets += 1
            slot.last_physics_step = diagnostics.physics_step
            return ParallelResetResult(
                env_id=slot.env_id,
                state=state,
                diagnostics=diagnostics,
                physics_step=diagnostics.physics_step,
            )
        except Exception as error:
            if saved_rng_state is not None:
                slot.rng.bit_generator.state = copy.deepcopy(saved_rng_state)
            raise _WorkerTaskError(self._failure(slot, error)) from error

    def _run_step(self, request: ParallelStepRequest) -> ParallelStepResult:
        slot = self._slots[request.env_id]
        try:
            slot.backend.apply_targets(request.targets_rad)
            result = slot.backend.step_substeps(request.substeps)
            slot.last_physics_step = result.diagnostics.physics_step
            return ParallelStepResult(
                env_id=slot.env_id,
                state=result.state,
                contacts=result.contacts,
                diagnostics=result.diagnostics,
                substeps_executed=result.substeps_executed,
                physics_step=result.diagnostics.physics_step,
                substep_trace=result.substep_trace,
            )
        except Exception as error:
            raise _WorkerTaskError(self._failure(slot, error)) from error

    def _run_actuator_control_ranges(self, env_id: int) -> ParallelActuatorControlRanges:
        slot = self._slots[env_id]
        try:
            return ParallelActuatorControlRanges(env_id, slot.backend.read_actuator_control_ranges())
        except Exception as error:
            raise _WorkerTaskError(self._failure(slot, error)) from error

    def _run_site_hinge_kinematics(
        self,
        env_id: int,
        site_name: str,
        joint_name: str,
    ) -> ParallelSiteHingeKinematics:
        slot = self._slots[env_id]
        try:
            value = slot.backend.read_site_hinge_kinematics(site_name, joint_name)
            return ParallelSiteHingeKinematics(env_id, value)
        except Exception as error:
            raise _WorkerTaskError(self._failure(slot, error)) from error

    def _run_arm_site_jacobian(self, env_id: int, site_name: str) -> ParallelArmSiteJacobian:
        slot = self._slots[env_id]
        try:
            positions, jacobian = slot.backend.read_arm_site_jacobian(site_name)
            return ParallelArmSiteJacobian(env_id, positions, jacobian)
        except Exception as error:
            raise _WorkerTaskError(self._failure(slot, error)) from error

    @staticmethod
    def _collect(
        futures: Sequence[tuple[int, Future[_ResultT]]],
    ) -> tuple[_ResultT, ...]:
        results: list[_ResultT] = []
        failures: list[WorkerFailure] = []
        for env_id, future in futures:
            try:
                results.append(future.result())
            except _WorkerTaskError as error:
                failures.append(error.failure)
            except Exception as error:
                failures.append(
                    WorkerFailure(
                        env_id=env_id,
                        physics_step=0,
                        exception_type=type(error).__name__,
                        message=str(error).strip() or repr(error),
                        config_fingerprint="unknown-worker-fingerprint",
                    )
                )
        if failures:
            raise ParallelWorkerError(failures)
        return tuple(sorted(results, key=lambda result: result.env_id))

    def reset(
        self,
        requests: Iterable[ParallelResetRequest] | None = None,
        *,
        reset_spec_factory: ResetSpecFactory | None = None,
    ) -> BatchResetResult:
        """Synchronously reset selected environments and publish after the barrier."""

        with self._operation_lock:
            self._require_open()
            source = (
                (ParallelResetRequest(env_id) for env_id in range(self._num_envs)) if requests is None else requests
            )
            ordered = self._validate_requests(source)
            futures = [
                (
                    request.env_id,
                    self._pool.submit(self._run_reset, request, reset_spec_factory),
                )
                for request in ordered
            ]
            results = self._collect(futures)
            self._generation += 1
            state = self._buffers.publish(
                self._generation,
                [result.env_id for result in results],
                [result.state for result in results],
            )
            return BatchResetResult(results=results, state=state)

    def step(self, requests: Iterable[ParallelStepRequest]) -> BatchStepResult:
        """Synchronously execute selected environments and publish after the barrier."""

        with self._operation_lock:
            self._require_open()
            ordered = self._validate_requests(requests)
            futures = [(request.env_id, self._pool.submit(self._run_step, request)) for request in ordered]
            results = self._collect(futures)
            self._generation += 1
            state = self._buffers.publish(
                self._generation,
                [result.env_id for result in results],
                [result.state for result in results],
            )
            return BatchStepResult(results=results, state=state)

    def read_actuator_control_ranges(
        self,
        env_ids: Iterable[int] | None = None,
    ) -> tuple[ParallelActuatorControlRanges, ...]:
        """Query independent models behind a barrier and return env-id order."""

        with self._operation_lock:
            self._require_open()
            ids = tuple(range(self._num_envs)) if env_ids is None else tuple(env_ids)
            ordered = self._validate_requests(ParallelResetRequest(env_id) for env_id in ids)
            futures = [
                (request.env_id, self._pool.submit(self._run_actuator_control_ranges, request.env_id))
                for request in ordered
            ]
            return self._collect(futures)

    def read_site_hinge_kinematics(
        self,
        site_name: str,
        joint_name: str,
        env_ids: Iterable[int] | None = None,
    ) -> tuple[ParallelSiteHingeKinematics, ...]:
        """Query one named site-hinge pair per independent model in env-id order."""

        with self._operation_lock:
            self._require_open()
            ids = tuple(range(self._num_envs)) if env_ids is None else tuple(env_ids)
            ordered = self._validate_requests(ParallelResetRequest(env_id) for env_id in ids)
            futures = [
                (
                    request.env_id,
                    self._pool.submit(
                        self._run_site_hinge_kinematics,
                        request.env_id,
                        site_name,
                        joint_name,
                    ),
                )
                for request in ordered
            ]
            return self._collect(futures)

    def read_arm_site_jacobian(
        self,
        site_name: str,
        env_ids: Iterable[int] | None = None,
    ) -> tuple[ParallelArmSiteJacobian, ...]:
        """Query arm positions and one site Jacobian per selected environment."""

        with self._operation_lock:
            self._require_open()
            ids = tuple(range(self._num_envs)) if env_ids is None else tuple(env_ids)
            ordered = self._validate_requests(ParallelResetRequest(env_id) for env_id in ids)
            futures = [
                (
                    request.env_id,
                    self._pool.submit(self._run_arm_site_jacobian, request.env_id, site_name),
                )
                for request in ordered
            ]
            return self._collect(futures)

    def rng_manifest(self) -> tuple[EnvironmentRngManifest, ...]:
        """Return a deterministic, JSON-safe snapshot of every RNG stream."""

        with self._operation_lock:
            self._require_open()
            return tuple(
                EnvironmentRngManifest(
                    env_id=slot.env_id,
                    root_seed=self._root_seed,
                    bit_generator=type(slot.rng.bit_generator).__name__,
                    state_json=json.dumps(
                        slot.rng.bit_generator.state,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    successful_resets=slot.successful_resets,
                )
                for slot in self._slots
            )

    def restore_rng_manifest(self, manifest: Iterable[EnvironmentRngManifest]) -> None:
        """Fail closed and restore all streams atomically before further work."""

        with self._operation_lock:
            self._require_open()
            entries = tuple(sorted(manifest, key=lambda entry: entry.env_id))
            if tuple(entry.env_id for entry in entries) != tuple(range(self._num_envs)):
                raise ValueError("RNG manifest must contain every env_id exactly once")
            parsed_states: list[Mapping[str, object]] = []
            for entry in entries:
                if entry.root_seed != self._root_seed:
                    raise ValueError("RNG manifest root seed mismatch")
                if entry.bit_generator != "PCG64":
                    raise ValueError("RNG manifest bit generator mismatch")
                value = json.loads(entry.state_json)
                if not isinstance(value, dict):
                    raise ValueError("RNG manifest state must decode to an object")
                parsed_states.append(cast(Mapping[str, object], value))
            original_states = [copy.deepcopy(slot.rng.bit_generator.state) for slot in self._slots]
            original_reset_counts = [slot.successful_resets for slot in self._slots]
            try:
                for slot, state, entry in zip(self._slots, parsed_states, entries, strict=True):
                    slot.rng.bit_generator.state = copy.deepcopy(state)
                    slot.successful_resets = entry.successful_resets
            except Exception:
                for slot, state, reset_count in zip(
                    self._slots,
                    original_states,
                    original_reset_counts,
                    strict=True,
                ):
                    slot.rng.bit_generator.state = state
                    slot.successful_resets = reset_count
                raise

    def close(self) -> None:
        """Wait for pending work, close every backend, then join worker threads."""

        with self._state_lock:
            if self._closed or self._closing:
                return
            self._closing = True
        failures: list[WorkerFailure] = []
        with self._operation_lock:
            for slot in self._slots:
                try:
                    slot.backend.close()
                except Exception as error:
                    failures.append(self._failure(slot, error))
            self._pool.shutdown(wait=True, cancel_futures=True)
            with self._state_lock:
                self._closed = True
                self._closing = False
        if failures:
            raise ParallelWorkerError(failures)

    def __enter__(self) -> ThreadPoolParallelExecutor:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "BatchResetResult",
    "BatchStepResult",
    "BatchedPhysicsState",
    "EnvironmentIdentity",
    "EnvironmentRngManifest",
    "ParallelExecutorClosedError",
    "ParallelExecutorError",
    "ParallelResetRequest",
    "ParallelResetResult",
    "ParallelStepRequest",
    "ParallelStepResult",
    "ParallelWorkerError",
    "ResetSpecFactory",
    "ThreadPoolParallelExecutor",
    "WorkerFailure",
]
