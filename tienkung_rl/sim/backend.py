"""Simulator-agnostic single-environment backend contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .state import (
    ActuatorControlRanges,
    ContactFrame,
    PhysicsState,
    PhysicsStepResult,
    ResetSpec,
    SimulationDiagnostics,
    SiteHingeKinematics,
)


@runtime_checkable
class EnvironmentBackend(Protocol):
    """Named physics boundary consumed by task and parallel execution layers."""

    def reset(self, reset_spec: ResetSpec) -> PhysicsState:
        """Restore one episode and return its finite initial state."""

    def apply_targets(self, targets_rad: NDArray[np.float64]) -> None:
        """Write exactly 7 right-arm plus 6 active-hand position targets."""

    def step_substeps(self, count: int) -> PhysicsStepResult:
        """Advance exactly ``count`` physical substeps and aggregate contacts."""

    def read_state(self) -> PhysicsState:
        """Return a read-only named state snapshot."""

    def read_contacts(self) -> ContactFrame:
        """Return contacts from the most recent substep group."""

    def read_warnings(self) -> SimulationDiagnostics:
        """Return cumulative warnings and finite-state diagnostics."""

    def read_actuator_control_ranges(self) -> ActuatorControlRanges:
        """Return immutable named actuator control bounds."""

    def read_site_hinge_kinematics(self, site_name: str, joint_name: str) -> SiteHingeKinematics:
        """Return one named site's Jacobian column for a scalar hinge."""

    def read_arm_site_jacobian(
        self,
        site_name: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return arm positions and a world-frame ``6 x 7`` site Jacobian."""

    def close(self) -> None:
        """Release this backend; repeated calls are harmless."""
