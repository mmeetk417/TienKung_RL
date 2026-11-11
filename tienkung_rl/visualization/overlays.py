"""Pure, read-only formatting of M12 task snapshots for the native viewer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from tienkung_rl.task.environment import TaskSnapshot


@dataclass(frozen=True, slots=True)
class OverlaySection:
    """One native-viewer overlay block."""

    title: str
    content: str


class OverlayTarget(Protocol):
    def set_texts(self, texts: object) -> None: ...


def _flag(value: bool) -> str:
    return "YES" if value else "no"


def _reasons(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


def build_overlay_sections(
    snapshot: TaskSnapshot,
    *,
    mode: str,
    realtime_factor: float,
) -> tuple[OverlaySection, ...]:
    """Render every required diagnostic without retaining or mutating state."""

    if mode not in {"policy", "random", "zero", "scripted"}:
        raise ValueError(f"unknown viewer mode {mode!r}")
    if not math.isfinite(realtime_factor) or realtime_factor < 0.0:
        raise ValueError("realtime_factor must be finite and non-negative")
    reward_lines = [f"step={snapshot.step_reward:+.5f}  episode={snapshot.episode_reward_total:+.5f}"]
    reward_lines.extend(f"{name}={value:+.5f}" for name, value in sorted(snapshot.reward_terms.items()))
    status = OverlaySection(
        title=f"reach_then_close [{mode}]",
        content=(
            f"stage={snapshot.stage_at_reset}  episode_step={snapshot.episode_step}\n"
            f"sim={snapshot.simulation_time_s:.3f}s  physics_step={snapshot.physics_step}\n"
            f"realtime={realtime_factor:.3f}x"
        ),
    )
    task = OverlaySection(
        title="Task / admission",
        content=(
            f"position_error={snapshot.position_error_m:.6f} m\n"
            f"orientation_error={math.degrees(snapshot.orientation_error_rad):.3f} deg\n"
            f"ready={_flag(snapshot.strict_ready)} ({snapshot.strict_ready_consecutive_steps} steps)\n"
            f"admission_now={_flag(snapshot.admission_instantaneous)}  "
            f"dwell={snapshot.admission_consecutive_steps}/{snapshot.admission_dwell_steps_required}\n"
            f"auto_admissible={_flag(snapshot.option_admissible)}\n"
            f"admission_failures={_reasons(snapshot.admission_failures)}"
        ),
    )
    option = OverlaySection(
        title="Automatic handoff / Option / outcome",
        content=(
            f"request={_flag(snapshot.trigger_requested)}  accepted={_flag(snapshot.trigger_accepted)} "
            f"rejected={_flag(snapshot.trigger_rejected)}\n"
            f"phase={snapshot.option_phase}  phase_step={snapshot.option_phase_step}\n"
            f"script={snapshot.option_script_steps} ({snapshot.option_script_progress:.3f}) "
            f"support={snapshot.option_support_progress:.3f}\n"
            f"Lift5cm={_flag(snapshot.lift_5cm)}  success={_flag(snapshot.success)} "
            f"unsafe={_flag(snapshot.unsafe)}\n"
            f"primary={snapshot.primary_termination_reason or 'none'}\n"
            f"auxiliary={_reasons(snapshot.secondary_termination_reasons)}"
        ),
    )
    diagnostics = OverlaySection(
        title="Contact / simulator diagnostics",
        content=(
            f"contacts={snapshot.contact_count}  max_force={snapshot.maximum_contact_force_n:.3f} N\n"
            f"max_impulse={snapshot.maximum_contact_impulse_n_s:.6f} N*s  "
            f"capacity={snapshot.maximum_contacts_seen}/{snapshot.contact_capacity}\n"
            f"object_displacement={snapshot.object_displacement_m:.6f} m  "
            f"tilt={math.degrees(snapshot.object_tilt_rad):.3f} deg\n"
            f"warnings={_reasons(snapshot.simulation_warning_names)}\n"
            f"capacity_exceeded={_flag(snapshot.contact_capacity_exceeded)}  "
            f"solver_anomaly={_flag(snapshot.solver_anomaly)}  "
            f"nonfinite={_flag(snapshot.nonfinite_state)}\n"
            f"context=env:{snapshot.env_id} physics_step:{snapshot.physics_step} "
            f"primary:{snapshot.primary_termination_reason or 'none'}"
        ),
    )
    rewards = OverlaySection("Reward terms (episode totals)", "\n".join(reward_lines))
    return (status, task, option, diagnostics, rewards)


def apply_native_overlays(target: OverlayTarget, sections: tuple[OverlaySection, ...]) -> None:
    """Copy pure sections into MuJoCo's top-left overlay grid."""

    import mujoco  # type: ignore[import-untyped]

    target.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "\n\n".join(section.title for section in sections),
            "\n\n".join(section.content for section in sections),
        )
    )


def clear_native_overlays(target: OverlayTarget) -> None:
    """Clear the structured top-left text without changing task state."""

    import mujoco

    target.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "",
            "",
        )
    )


__all__ = ["OverlaySection", "apply_native_overlays", "build_overlay_sections", "clear_native_overlays"]
