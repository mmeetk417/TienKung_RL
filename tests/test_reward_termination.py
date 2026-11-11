"""M10 pure reward, termination, and inherited-semantics tests."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from tienkung_rl.config import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig, RewardConfig
from tienkung_rl.task.reward import (
    REWARD_TERM_NAMES,
    RewardBreakdown,
    RewardEpisodeState,
    RewardEvents,
    RewardNonFiniteError,
    RewardState,
    RewardTermSwitches,
    compute_reward,
)
from tienkung_rl.task.termination import (
    TERMINATION_REASON_NAMES,
    InheritedTerminationInput,
    OutcomeClassifier,
    OutcomeEpisodeState,
    OutcomeInput,
    TerminationReason,
    classify_outcome,
    evaluate_inherited_termination_signals,
)
from tienkung_rl.telemetry.metrics import REWARD_TERMS, TERMINATION_REASONS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> ResolvedRunConfig:
    return load_run_config(ROOT / "configs/reach_then_close.yaml")


def _state(**overrides: object) -> RewardState:
    values: dict[str, object] = {
        "position_error_m": 0.04,
        "orientation_error_quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
        "arm_joint_velocities_rad_s": (0.0,) * 7,
        "arm_action": (0.0,) * 7,
        "object_displacement_m": 0.0,
    }
    values.update(overrides)
    return RewardState(**values)  # type: ignore[arg-type]


def _reward(
    config: ResolvedRunConfig,
    *,
    before: RewardState | None = None,
    after: RewardState | None = None,
    events: RewardEvents | None = None,
    state: RewardEpisodeState | None = None,
    switches: RewardTermSwitches | None = None,
) -> RewardBreakdown:
    return compute_reward(
        before or _state(),
        after or _state(),
        events or RewardEvents(),
        state or RewardEpisodeState(),
        config.task.reward,
        switches=switches,
    ).breakdown


def _only(term: str) -> RewardTermSwitches:
    return RewardTermSwitches(**{name: name == term for name in REWARD_TERM_NAMES})


def _quat_for_angle(angle_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(angle_rad / 2.0), math.sin(angle_rad / 2.0), 0.0, 0.0)


def _module_source(member: object) -> str:
    module = inspect.getmodule(member)
    assert module is not None
    return inspect.getsource(module)


def test_stable_reward_and_termination_names_match_telemetry() -> None:
    assert (*REWARD_TERM_NAMES, "total") == REWARD_TERMS
    assert TERMINATION_REASON_NAMES == TERMINATION_REASONS
    assert len(REWARD_TERM_NAMES) == len(set(REWARD_TERM_NAMES))
    assert len(TERMINATION_REASON_NAMES) == len(set(TERMINATION_REASON_NAMES))


def test_reward_inputs_are_minimal_pure_data_and_no_simulator_dependency() -> None:
    assert tuple(field.name for field in dataclasses.fields(RewardState)) == (
        "position_error_m",
        "orientation_error_quaternion_wxyz",
        "arm_joint_velocities_rad_s",
        "arm_action",
        "object_displacement_m",
    )
    source = _module_source(compute_reward)
    assert "mujoco" not in source
    assert "legacy" not in source


@pytest.mark.parametrize(
    ("before_error", "after_error", "expected_sign"),
    ((0.06, 0.02, 1), (0.02, 0.06, -1), (0.04, 0.04, -1)),
)
def test_position_potential_progress_is_discount_consistent(
    config: ResolvedRunConfig,
    before_error: float,
    after_error: float,
    expected_sign: int,
) -> None:
    value = _reward(
        config,
        before=_state(position_error_m=before_error),
        after=_state(position_error_m=after_error),
        switches=_only("position_progress"),
    ).position_progress
    before_phi = math.exp(-((before_error / config.task.reward.position_potential_scale_m) ** 2))
    after_phi = math.exp(-((after_error / config.task.reward.position_potential_scale_m) ** 2))
    assert value == pytest.approx(
        config.task.reward.position_potential_weight * (config.task.reward.potential_discount * after_phi - before_phi)
    )
    assert math.copysign(1.0, value) == expected_sign


def test_orientation_progress_and_quaternion_double_cover(config: ResolvedRunConfig) -> None:
    before = _state(orientation_error_quaternion_wxyz=_quat_for_angle(0.30))
    next_quaternion = _quat_for_angle(0.10)
    before_quaternion = before.orientation_error_quaternion_wxyz
    positive = _reward(
        config,
        before=before,
        after=_state(orientation_error_quaternion_wxyz=next_quaternion),
        switches=_only("orientation_progress"),
    )
    negative_cover = _reward(
        config,
        before=replace(
            before,
            orientation_error_quaternion_wxyz=(
                -before_quaternion[0],
                -before_quaternion[1],
                -before_quaternion[2],
                -before_quaternion[3],
            ),
        ),
        after=_state(
            orientation_error_quaternion_wxyz=(
                -next_quaternion[0],
                -next_quaternion[1],
                -next_quaternion[2],
                -next_quaternion[3],
            )
        ),
        switches=_only("orientation_progress"),
    )
    assert positive.orientation_progress > 0.0
    assert negative_cover.orientation_progress == positive.orientation_progress


def test_basic_penalty_formulas_and_zero_inputs(config: ResolvedRunConfig) -> None:
    reward = config.task.reward
    baseline = _state()
    changed = _state(
        arm_joint_velocities_rad_s=(2.0,) * 7,
        arm_action=(1.0,) * 7,
        object_displacement_m=0.01,
    )
    breakdown = _reward(config, before=baseline, after=changed)
    assert breakdown.step_penalty == -reward.step_penalty
    assert breakdown.joint_velocity_penalty == -reward.joint_velocity_penalty_weight * 4.0
    assert breakdown.action_rate_penalty == -reward.action_rate_penalty_weight
    assert breakdown.object_displacement_penalty == -reward.object_displacement_penalty_weight * 0.25
    assert breakdown.unsafe_object_displacement_penalty == 0.0
    zero = _reward(config, before=baseline, after=baseline)
    assert zero.joint_velocity_penalty == 0.0
    assert zero.action_rate_penalty == 0.0
    assert zero.object_displacement_penalty == 0.0
    assert zero.unsafe_object_displacement_penalty == 0.0


def test_unsafe_displacement_is_strictly_above_configured_boundary(config: ResolvedRunConfig) -> None:
    threshold = config.task.reward.object_displacement_scale_m
    at_limit = _reward(config, after=_state(object_displacement_m=threshold))
    above = _reward(config, after=_state(object_displacement_m=math.nextafter(threshold, math.inf)))
    assert at_limit.unsafe_object_displacement_penalty == 0.0
    assert above.unsafe_object_displacement_penalty == -config.task.reward.unsafe_object_displacement_penalty


@pytest.mark.parametrize(
    ("term", "events"),
    (
        ("accepted_trigger", RewardEvents(trigger_requested=True, trigger_accepted=True)),
        ("lift_5cm", RewardEvents(reach_phase=False, lift_5cm_event=True)),
        ("complete_success", RewardEvents(reach_phase=False, complete_success=True)),
        ("script_failure", RewardEvents(reach_phase=False, script_failure=True)),
    ),
)
def test_event_reward_weight_sign_and_independent_disable(
    config: ResolvedRunConfig,
    term: str,
    events: RewardEvents,
) -> None:
    enabled = _reward(config, events=events, switches=_only(term))
    assert getattr(enabled, term) != 0.0
    expected_positive = term in {"accepted_trigger", "lift_5cm", "complete_success"}
    assert (getattr(enabled, term) > 0.0) is expected_positive
    disabled = _reward(config, events=events, switches=RewardTermSwitches(**{term: False}))
    assert getattr(disabled, term) == 0.0


@pytest.mark.parametrize("term", REWARD_TERM_NAMES)
def test_every_reward_term_can_be_disabled_independently(config: ResolvedRunConfig, term: str) -> None:
    before = _state(position_error_m=0.07, arm_action=(-1.0,) * 7)
    after = _state(
        position_error_m=0.01,
        orientation_error_quaternion_wxyz=_quat_for_angle(0.05),
        arm_joint_velocities_rad_s=(1.0,) * 7,
        arm_action=(1.0,) * 7,
        object_displacement_m=0.03,
    )
    events = RewardEvents(
        option_admissible=True,
        trigger_requested=True,
        trigger_accepted=True,
        lift_5cm_event=True,
        complete_success=True,
    )
    enabled = _reward(config, before=before, after=after, events=events)
    disabled = _reward(
        config,
        before=before,
        after=after,
        events=events,
        switches=RewardTermSwitches(**{term: False}),
    )
    assert getattr(disabled, term) == 0.0
    for other in REWARD_TERM_NAMES:
        if other != term:
            assert getattr(disabled, other) == getattr(enabled, other)


def test_event_latches_pay_once_and_obsolete_manual_trigger_penalties_are_zero(config: ResolvedRunConfig) -> None:
    first_events = RewardEvents(
        reach_phase=False,
        trigger_requested=True,
        trigger_accepted=True,
        lift_5cm_event=True,
        complete_success=True,
    )
    first = compute_reward(_state(), _state(), first_events, RewardEpisodeState(), config.task.reward)
    second = compute_reward(_state(), _state(), first_events, first.episode_state, config.task.reward)
    assert first.breakdown.accepted_trigger == 3.0
    assert first.breakdown.lift_5cm == 10.0
    assert first.breakdown.complete_success == 50.0
    assert second.breakdown.accepted_trigger == 0.0
    assert second.breakdown.lift_5cm == 0.0
    assert second.breakdown.complete_success == 0.0
    rejected = RewardEvents(trigger_requested=True, trigger_rejected=True)
    rejection1 = compute_reward(_state(), _state(), rejected, RewardEpisodeState(), config.task.reward)
    rejection2 = compute_reward(_state(), _state(), rejected, rejection1.episode_state, config.task.reward)
    assert rejection1.breakdown.rejected_trigger == rejection2.breakdown.rejected_trigger == 0.0
    waiting = compute_reward(
        _state(),
        _state(),
        RewardEvents(option_admissible=True),
        RewardEpisodeState(),
        config.task.reward,
    )
    assert waiting.breakdown.admissible_wait == 0.0


def test_ready_appearance_duration_and_latch_are_strictly_reward_neutral(config: ResolvedRunConfig) -> None:
    baseline = _reward(config, events=RewardEvents())
    variants = (
        RewardEvents(strict_ready=True),
        RewardEvents(strict_ready=True, newly_strict_ready=True),
        RewardEvents(strict_ready=True, strict_ready_ever=True),
        RewardEvents(strict_ready=False, strict_ready_ever=True),
    )
    for events in variants:
        assert _reward(config, events=events) == baseline
    assert "ready" not in REWARD_TERM_NAMES
    assert config.task.reward.ready_reward == 0.0


def test_reward_total_is_exact_named_sum_and_mapping_is_immutable(config: ResolvedRunConfig) -> None:
    breakdown = _reward(
        config,
        before=_state(arm_action=(-0.5,) * 7),
        after=_state(position_error_m=0.01, arm_action=(0.5,) * 7),
        events=RewardEvents(trigger_requested=True, trigger_accepted=True),
    )
    assert breakdown.total == math.fsum(breakdown.terms.values())
    assert tuple(breakdown.terms) == REWARD_TERM_NAMES
    with pytest.raises(TypeError):
        breakdown.terms["step_penalty"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="does not equal"):
        RewardBreakdown(**breakdown.terms, total=breakdown.total + 1.0)


@pytest.mark.parametrize("term", REWARD_TERM_NAMES)
def test_every_nonfinite_named_reward_fails_with_full_breakdown_context(term: str) -> None:
    values = {name: 0.0 for name in REWARD_TERM_NAMES}
    values[term] = math.nan
    with pytest.raises(RewardNonFiniteError) as raised:
        RewardBreakdown(**values, total=math.nan)
    assert tuple(raised.value.term_context) == (*REWARD_TERM_NAMES, "total")
    assert term in str(raised.value)


@pytest.mark.parametrize(
    "after",
    (
        _state(position_error_m=math.nan),
        _state(orientation_error_quaternion_wxyz=(math.inf, 0.0, 0.0, 0.0)),
        _state(arm_joint_velocities_rad_s=(math.nan,) + (0.0,) * 6),
        _state(arm_action=(math.inf,) + (0.0,) * 6),
        _state(object_displacement_m=math.inf),
    ),
)
def test_nonfinite_reward_inputs_fail_immediately_with_named_context(
    config: ResolvedRunConfig,
    after: RewardState,
) -> None:
    with pytest.raises(RewardNonFiniteError) as raised:
        compute_reward(_state(), after, RewardEvents(), RewardEpisodeState(), config.task.reward)
    assert tuple(raised.value.term_context) == (*REWARD_TERM_NAMES, "total")


@pytest.mark.parametrize(
    "before",
    (
        _state(arm_joint_velocities_rad_s=(math.nan,) + (0.0,) * 6),
        _state(object_displacement_m=math.nan),
    ),
)
def test_nonfinite_unused_before_fields_are_not_silently_ignored(
    config: ResolvedRunConfig,
    before: RewardState,
) -> None:
    with pytest.raises(RewardNonFiniteError, match=r"before\.") as raised:
        compute_reward(
            before,
            _state(),
            RewardEvents(),
            RewardEpisodeState(),
            config.task.reward,
            switches=RewardTermSwitches(**{name: False for name in REWARD_TERM_NAMES}),
        )
    assert tuple(raised.value.term_context) == (*REWARD_TERM_NAMES, "total")


def test_reward_weights_and_scales_are_read_from_config(config: ResolvedRunConfig) -> None:
    reward = config.task.reward
    assert reward.version == "scratch_reward_auto_trigger_v4"
    assert reward.position_potential_scale_m == 0.08
    assert reward.orientation_potential_scale_rad == 0.35
    assert reward.object_displacement_scale_m == 0.020
    source = _module_source(compute_reward)
    for forbidden_literal in ("0.08", "0.35", "0.005", "0.020"):
        assert forbidden_literal not in source


def _contact_input(config: ResolvedRunConfig, **overrides: object) -> InheritedTerminationInput:
    values: dict[str, object] = {
        "object_center_z_m": 1.0,
        "table_top_z_m": 0.9,
        "object_position_xy_in_environment_frame_m": (0.0, 0.0),
        "contact_force_by_group_n": {
            config.task.termination.thumb_contact_group: 0.0,
            config.task.termination.opposing_contact_group: 0.0,
            config.task.termination.lift_support_contact_group: 0.0,
        },
    }
    values.update(overrides)
    return InheritedTerminationInput(**values)  # type: ignore[arg-type]


def test_static_fixture_locks_inherited_semantics_without_importing_legacy(config: ResolvedRunConfig) -> None:
    fixture_path = ROOT / "tests/fixtures/scratch_reward_termination_reference_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    reward = fixture["reward"]
    termination = fixture["termination"]
    for name, value in reward.items():
        assert getattr(config.task.reward, name) == value
    for name, value in termination.items():
        assert getattr(config.task.termination, name) == value
    source = _module_source(evaluate_inherited_termination_signals)
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert "import legacy" not in source
    assert "from legacy" not in source
    imports = (node for node in ast.walk(ast.parse(test_source)) if isinstance(node, (ast.Import, ast.ImportFrom)))
    assert all(
        not any(alias.name == "legacy" or alias.name.startswith("legacy.") for alias in node.names) for node in imports
    )


def test_inherited_drop_far_and_contact_boundaries(config: ResolvedRunConfig) -> None:
    termination = config.task.termination
    at_drop = evaluate_inherited_termination_signals(
        _contact_input(config, object_center_z_m=0.9 - termination.drop_below_table_top_m),
        termination,
    )
    below_drop = evaluate_inherited_termination_signals(
        _contact_input(
            config,
            object_center_z_m=math.nextafter(0.9 - termination.drop_below_table_top_m, -math.inf),
        ),
        termination,
    )
    at_far = evaluate_inherited_termination_signals(
        _contact_input(config, object_position_xy_in_environment_frame_m=(termination.far_xy_distance_m, 0.0)),
        termination,
    )
    beyond_far = evaluate_inherited_termination_signals(
        _contact_input(
            config,
            object_position_xy_in_environment_frame_m=(
                math.nextafter(termination.far_xy_distance_m, math.inf),
                0.0,
            ),
        ),
        termination,
    )
    contact = evaluate_inherited_termination_signals(
        _contact_input(
            config,
            contact_force_by_group_n={
                termination.thumb_contact_group: termination.bilateral_contact_force_n,
                termination.opposing_contact_group: termination.bilateral_contact_force_n,
                termination.lift_support_contact_group: termination.bilateral_contact_force_n,
            },
        ),
        termination,
    )
    assert not at_drop.dropped and below_drop.dropped
    assert not at_far.object_far and beyond_far.object_far
    assert contact.bilateral_contact and contact.lift_supported


def _safe_grasp_step(step_index: int, **overrides: object) -> OutcomeInput:
    values: dict[str, object] = {
        "step_index": step_index,
        "option_accepted": step_index == 0,
        "object_lift_height_m": 0.05,
        "thumb_contact_force_n": 0.20,
        "opposing_contact_force_n": 0.20,
        "lift_support_force_n": 0.20,
    }
    values.update(overrides)
    return OutcomeInput(**values)  # type: ignore[arg-type]


def test_lift_event_is_one_shot_and_complete_success_is_full_conjunction(config: ResolvedRunConfig) -> None:
    classifier = OutcomeClassifier(config.task.termination)
    results = [classifier.step(_safe_grasp_step(step)) for step in range(4)]
    final = classifier.step(_safe_grasp_step(4, completed_lift_hold=True))
    assert [result.lift_5cm_event for result in results] == [False, False, True, False]
    assert not any(result.success for result in results)
    assert final.lift_5cm and not final.lift_5cm_event
    assert final.success and final.done and not final.unsafe
    assert final.primary_reason is None and final.secondary_reasons == ()
    assert final.episode_state.option_accepted_ever
    assert final.episode_state.bilateral_persistent_ever
    assert final.episode_state.lift_support_persistent_ever
    assert final.episode_state.completed_lift_hold_ever
    assert final.episode_state.all_states_finite


@pytest.mark.parametrize(
    "missing",
    (
        "option_accepted",
        "lift_5cm",
        "completed_lift_hold",
        "bilateral_contact",
        "lift_support",
    ),
)
def test_each_success_conjunct_is_required(config: ResolvedRunConfig, missing: str) -> None:
    classifier = OutcomeClassifier(config.task.termination)
    for step in range(5):
        overrides: dict[str, object] = {"completed_lift_hold": step == 4}
        if missing == "option_accepted":
            overrides["option_accepted"] = False
        if missing == "lift_5cm":
            overrides["object_lift_height_m"] = 0.049
        if missing == "completed_lift_hold":
            overrides["completed_lift_hold"] = False
        if missing == "bilateral_contact":
            overrides["thumb_contact_force_n"] = 0.19
        if missing == "lift_support":
            overrides["lift_support_force_n"] = 0.19
        result = classifier.step(_safe_grasp_step(step, **overrides))
    assert not result.success


def test_lift5cm_is_retained_after_later_danger_but_success_is_revoked(config: ResolvedRunConfig) -> None:
    classifier = OutcomeClassifier(config.task.termination)
    for step in range(5):
        result = classifier.step(_safe_grasp_step(step, completed_lift_hold=step == 4))
    assert result.success and result.lift_5cm
    dangerous = classifier.step(_safe_grasp_step(5, toppled=True))
    assert dangerous.lift_5cm and not dangerous.lift_5cm_event
    assert dangerous.unsafe and not dangerous.success and dangerous.done
    assert dangerous.primary_reason is TerminationReason.TOPPLE


def test_same_step_priority_and_all_auxiliary_reasons(config: ResolvedRunConfig) -> None:
    result = classify_outcome(
        OutcomeInput(
            step_index=7,
            object_lift_height_m=math.nan,
            illegal_contact=True,
            object_far=True,
            excessive_displacement=True,
            toppled=True,
            dropped=True,
            simulation_error=True,
            script_failed=True,
            timed_out=True,
        ),
        OutcomeEpisodeState(),
        config.task.termination,
    )
    assert result.primary_reason is TerminationReason.NONFINITE_OR_SIMULATION_ERROR
    assert result.secondary_reasons == (
        TerminationReason.ILLEGAL_CONTACT,
        TerminationReason.EXCESSIVE_DISPLACEMENT,
        TerminationReason.TOPPLE,
        TerminationReason.DROP,
        TerminationReason.SCRIPT_FAILURE,
        TerminationReason.TIMEOUT,
    )
    assert result.secondary_reason_names == (
        "illegal_contact",
        "excessive_displacement",
        "topple",
        "drop",
        "script_failure",
        "timeout",
    )
    assert result.unsafe and not result.success and result.done


def test_earliest_failure_wins_across_steps_and_later_reasons_are_retained(config: ResolvedRunConfig) -> None:
    classifier = OutcomeClassifier(config.task.termination)
    first = classifier.step(OutcomeInput(step_index=2, timed_out=True))
    second = classifier.step(OutcomeInput(step_index=3, illegal_contact=True, toppled=True))
    third = classifier.step(OutcomeInput(step_index=4, object_lift_height_m=math.inf, dropped=True))
    assert first.primary_reason is TerminationReason.TIMEOUT
    assert second.primary_reason is TerminationReason.TIMEOUT
    assert third.primary_reason is TerminationReason.TIMEOUT
    assert third.secondary_reasons == (
        TerminationReason.ILLEGAL_CONTACT,
        TerminationReason.TOPPLE,
        TerminationReason.NONFINITE_OR_SIMULATION_ERROR,
        TerminationReason.DROP,
    )
    assert third.unsafe and not third.success


@pytest.mark.parametrize(
    ("overrides", "reason", "unsafe"),
    (
        ({"state_finite": False}, TerminationReason.NONFINITE_OR_SIMULATION_ERROR, True),
        ({"object_lift_height_m": math.nan}, TerminationReason.NONFINITE_OR_SIMULATION_ERROR, True),
        ({"simulation_error": True}, TerminationReason.NONFINITE_OR_SIMULATION_ERROR, True),
        ({"illegal_contact": True}, TerminationReason.ILLEGAL_CONTACT, True),
        ({"object_far": True}, TerminationReason.EXCESSIVE_DISPLACEMENT, True),
        ({"toppled": True}, TerminationReason.TOPPLE, True),
        ({"dropped": True}, TerminationReason.DROP, True),
        ({"script_failed": True}, TerminationReason.SCRIPT_FAILURE, False),
        ({"timed_out": True}, TerminationReason.TIMEOUT, False),
    ),
)
def test_each_failure_reason_terminates_with_stable_classification(
    config: ResolvedRunConfig,
    overrides: dict[str, object],
    reason: TerminationReason,
    unsafe: bool,
) -> None:
    result = classify_outcome(
        OutcomeInput(step_index=0, **overrides),  # type: ignore[arg-type]
        OutcomeEpisodeState(),
        config.task.termination,
    )
    assert result.done and not result.success
    assert result.primary_reason is reason
    assert result.primary_reason_name == reason.value
    assert result.unsafe is unsafe


def test_relative_object_displacement_does_not_terminate_as_unsafe(
    config: ResolvedRunConfig,
) -> None:
    result = classify_outcome(
        OutcomeInput(step_index=0, excessive_displacement=True),
        OutcomeEpisodeState(),
        config.task.termination,
    )
    assert not result.done
    assert not result.unsafe
    assert result.primary_reason is None


def test_outcome_classifier_reset_clears_all_latches(config: ResolvedRunConfig) -> None:
    classifier = OutcomeClassifier(config.task.termination)
    classifier.step(OutcomeInput(step_index=0, option_accepted=True, toppled=True))
    assert classifier.state.option_accepted_ever and classifier.state.reason_first_steps
    assert classifier.reset() == OutcomeEpisodeState()


def test_outcome_input_order_and_contact_boundaries_fail_closed(config: ResolvedRunConfig) -> None:
    classifier = OutcomeClassifier(config.task.termination)
    below = math.nextafter(config.task.termination.bilateral_contact_force_n, -math.inf)
    first = classifier.step(
        OutcomeInput(
            step_index=1,
            thumb_contact_force_n=below,
            opposing_contact_force_n=config.task.termination.bilateral_contact_force_n,
        )
    )
    assert not first.bilateral_contact
    with pytest.raises(ValueError, match="strictly increasing"):
        classifier.step(OutcomeInput(step_index=0))


def test_reward_config_nonfinite_reports_context(config: ResolvedRunConfig) -> None:
    invalid: RewardConfig = replace(config.task.reward, accepted_trigger_reward=math.nan)
    with pytest.raises(RewardNonFiniteError, match="accepted_trigger_reward") as raised:
        compute_reward(_state(), _state(), RewardEvents(), RewardEpisodeState(), invalid)
    assert "accepted_trigger_reward" in raised.value.term_context
