"""M18 telemetry schema, aggregation, output, and run-artifact tests."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from tienkung_rl.config import DEFAULT_CONFIG_PATH, load_run_config
from tienkung_rl.telemetry import (
    ADMISSION_FAILURE_REASONS,
    METRIC_SCHEMA,
    ConditionalAggregator,
    CurriculumWindowEvent,
    DiagnosticEvent,
    DiagnosticKind,
    EpisodeEvent,
    NonFiniteTelemetryError,
    PPOUpdateEvent,
    ReportMetadata,
    RunEvent,
    StepEvent,
    TelemetryAggregator,
    TelemetryDiagnostic,
    TelemetrySnapshot,
    TensorBoardTelemetryWriter,
    finalize_run_manifest,
    prepare_run_directory,
    render_terminal_summary,
    validate_run_directory,
    write_csv_report,
    write_final_reports,
    write_json_report,
)
from tienkung_rl.telemetry.tensorboard import TENSORBOARD_TAGS


def _step(*, env_id: int, accepted: bool) -> StepEvent:
    return StepEvent(
        iteration=3,
        env_id=env_id,
        policy_step=100 + env_id,
        physics_steps=4,
        actions=(-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0),
        action_clipped=(True, False, False, False, False, False, True),
        option_admissible=accepted,
        trigger_requested=True,
        trigger_accepted=accepted,
        trigger_rejected=not accepted,
        strict_ready=accepted,
        admission_failures=() if accepted else ("position_error", "object_tilt"),
        accepted_position_error_m=0.01 if accepted else None,
        accepted_orientation_error_rad=0.1 if accepted else None,
        contact_count=4 + env_id,
        maximum_contact_force_n=12.0 + env_id,
        maximum_contact_impulse_n_s=0.02 + 0.01 * env_id,
        object_tilt_rad=0.05 + 0.01 * env_id,
        object_displacement_m=0.005 + 0.001 * env_id,
    )


def _episode(*, env_id: int, stage: str, dangerous: bool) -> EpisodeEvent:
    rewards = {
        "position_progress": 2.0 + env_id,
        "complete_success": 0.0 if dangerous else 50.0,
        "total": -25.0 if dangerous else 52.0 + env_id,
    }
    samples = {
        "cylinder_radius_m": 0.012 + 0.001 * env_id,
        "cylinder_height_m": 0.08 + 0.01 * env_id,
        "cylinder_mass_kg": 0.10 + 0.01 * env_id,
        "hand_sliding_friction": 0.7 + 0.1 * env_id,
        "table_sliding_friction": 0.8 + 0.1 * env_id,
    }
    return EpisodeEvent(
        iteration=3,
        env_id=env_id,
        episode_length_steps=200 + env_id,
        stage_at_reset=stage,
        strict_ready_ever=not dangerous,
        lift_5cm=True,
        complete_success=not dangerous,
        dangerous=dangerous,
        primary_termination_reason="topple" if dangerous else None,
        auxiliary_termination_reasons=("excessive_displacement",) if dangerous else (),
        reward_totals=rewards,
        randomization_samples=samples,
    )


def _aggregated_snapshot() -> TelemetrySnapshot:
    aggregator = TelemetryAggregator()
    aggregator.ingest(_step(env_id=0, accepted=True))
    aggregator.ingest(_step(env_id=1, accepted=False))
    aggregator.ingest(
        PPOUpdateEvent(
            iteration=3,
            policy_loss=-0.2,
            value_loss=0.3,
            entropy=0.4,
            kl=0.01,
            total_loss=0.11,
            gradient_norm=0.5,
            gradients_finite=True,
            log_probability=-1.2,
        )
    )
    # Deliberately complete env 2 before env 0 to exercise asynchronous episodes.
    aggregator.ingest(_episode(env_id=2, stage="S0", dangerous=True))
    aggregator.ingest(_episode(env_id=0, stage="S1", dangerous=False))
    aggregator.ingest(
        CurriculumWindowEvent(
            iteration=3,
            stage="S1",
            completed_episodes=4096,
            target_episodes=4096,
            promoted=False,
            stalled=True,
            consecutive_passing_windows=0,
        )
    )
    aggregator.ingest(
        RunEvent(
            iteration=3,
            total_steps=262_144,
            steps_per_second=4096.0,
            sampling_seconds=50.0,
            learning_seconds=10.0,
            total_seconds=64.0,
            eta_seconds=3600.0,
            memory_rss_bytes=2_000_000_000,
            cpu_percent=1200.0,
            num_envs=128,
            worker_threads=16,
            physics_steps=1_048_576,
            policy_steps=262_144,
            sim_steps_per_second=16_384.0,
            realtime_factor=20.48,
            fixed_torsional_friction=0.005,
            fixed_rolling_friction=0.0001,
            randomization_version="cylinder_randomization_v1",
            bottleneck="contact broad phase",
        )
    )
    aggregator.ingest(
        DiagnosticEvent(
            3,
            TelemetryDiagnostic(
                DiagnosticKind.MUJOCO,
                "constraint warning",
                count=2,
                env_id=7,
                policy_step=99,
            ),
        )
    )
    aggregator.ingest(DiagnosticEvent(3, TelemetryDiagnostic(DiagnosticKind.WORKER, "worker 3 failed")))
    aggregator.ingest(DiagnosticEvent(3, TelemetryDiagnostic(DiagnosticKind.CAPACITY, "contact buffer full", count=3)))
    return aggregator.snapshot(3)


class _FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.text: list[tuple[str, str, int]] = []
        self.flushes = 0
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        self.text.append((tag, text_string, global_step))

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


def test_schema_is_complete_stable_and_has_unique_tensorboard_tags() -> None:
    required = {
        "throughput.steps_per_second",
        "ppo.log_probability",
        "action.rms",
        "admission.accepted_position_error_m",
        "outcome.complete_success_rate",
        "curriculum.window_progress",
        "randomization.cylinder_mass_kg.std",
        "contact.maximum_impulse_n_s",
        "warning.solver.count",
        "system.realtime_factor",
    }
    assert required <= set(METRIC_SCHEMA)
    assert len(TENSORBOARD_TAGS) == len(METRIC_SCHEMA) == len(set(TENSORBOARD_TAGS.values()))
    for reason in ADMISSION_FAILURE_REASONS:
        assert f"admission.failure.{reason}_rate" in METRIC_SCHEMA


def test_conditional_aggregator_empty_single_and_multiple_samples() -> None:
    aggregate = ConditionalAggregator()
    aggregate.add(99.0, condition=False)
    assert aggregate.result().value == 0.0
    assert aggregate.result().count == aggregate.result().denominator == 0
    assert math.isfinite(aggregate.result().value)
    aggregate.add(2.5)
    assert aggregate.result().value == 2.5
    assert aggregate.result().numerator == 2.5
    assert aggregate.result().count == 1
    aggregate.add(7.5)
    assert aggregate.result().value == 5.0
    assert aggregate.result().count == 2
    with pytest.raises(ValueError, match="finite"):
        aggregate.add(float("nan"))


def test_aggregation_contract_and_async_stage_at_reset_counts() -> None:
    snapshot = _aggregated_snapshot()
    assert set(snapshot.metrics) == set(METRIC_SCHEMA)
    assert snapshot.metrics["trigger.accept_rate"].value == pytest.approx(0.5)
    assert snapshot.metrics["admission.failure.position_error_rate"].value == pytest.approx(0.5)
    assert snapshot.metrics["contact.maximum_force_n"].value == pytest.approx(13.0)
    assert snapshot.metrics["outcome.lift_5cm_rate"].value == 1.0
    assert snapshot.metrics["outcome.dangerous_rate"].value == pytest.approx(0.5)
    assert snapshot.metrics["curriculum.episodes_at_reset.S0"].value == 1.0
    assert snapshot.metrics["curriculum.episodes_at_reset.S1"].value == 1.0
    assert snapshot.metrics["curriculum.episodes_at_reset.S2"].value == 0.0
    assert snapshot.metrics["curriculum.episodes_at_reset.S2"].count == 2
    assert snapshot.metrics["warning.mujoco.count"].value == 2.0
    assert snapshot.metrics["warning.worker.count"].value == 1.0
    assert snapshot.metrics["warning.capacity.count"].value == 3.0
    assert snapshot.randomization_version == "cylinder_randomization_v1"
    assert all(math.isfinite(item.value) for item in snapshot.metrics.values())


def test_empty_conditions_remain_zero_count_and_finite() -> None:
    aggregator = TelemetryAggregator()
    no_request = _step(env_id=0, accepted=False)
    no_request = StepEvent(
        **{
            **no_request.__dict__,
            "trigger_requested": False,
            "trigger_rejected": False,
            "option_admissible": False,
        }
    )
    aggregator.record_step(no_request)
    snapshot = aggregator.snapshot(3)
    for name in (
        "trigger.accept_rate",
        "trigger.reject_rate",
        "admission.accepted_position_error_m",
        "admission.accepted_orientation_error_rad",
    ):
        assert snapshot.metrics[name].value == 0.0
        assert snapshot.metrics[name].count == 0
        assert math.isfinite(snapshot.metrics[name].value)


def test_nonfinite_is_retained_then_fails_without_nan_output() -> None:
    aggregator = TelemetryAggregator()
    event = _step(env_id=0, accepted=True)
    bad = StepEvent(**{**event.__dict__, "actions": (float("nan"), *event.actions[1:])})
    with pytest.raises(NonFiniteTelemetryError, match="actions"):
        aggregator.record_step(bad)
    snapshot = aggregator.snapshot(3)
    assert snapshot.metrics["warning.nonfinite.count"].value == 1.0
    assert snapshot.diagnostics[0].kind is DiagnosticKind.NONFINITE
    assert all(math.isfinite(item.value) for item in snapshot.metrics.values())


def test_terminal_tensorboard_json_and_csv_have_identical_numbers(tmp_path: Path) -> None:
    snapshot = _aggregated_snapshot()
    config = load_run_config(DEFAULT_CONFIG_PATH)
    metadata = ReportMetadata.from_config(config)
    terminal = render_terminal_summary(snapshot)

    fake = _FakeWriter()
    tensorboard = TensorBoardTelemetryWriter(writer=fake)
    emitted = tensorboard.write_iteration(snapshot, metadata)
    assert fake.flushes == 1
    assert len(fake.scalars) == 2 * len(METRIC_SCHEMA)
    assert json.loads(fake.text[0][1])["config_fingerprint"] == metadata.config_fingerprint

    json_path = write_json_report(tmp_path / "final.json", snapshot, metadata)
    csv_path = write_csv_report(tmp_path / "final.csv", snapshot, metadata)
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    json_values = {row["name"]: row["value"] for row in json_payload["metrics"]}
    json_counts = {row["name"]: row["count"] for row in json_payload["metrics"]}
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_values = {row["name"]: float(row["value"]) for row in csv_rows}
    csv_counts = {row["name"]: int(row["count"]) for row in csv_rows}

    expected_values = {name: item.value for name, item in snapshot.metrics.items()}
    expected_counts = {name: item.count for name, item in snapshot.metrics.items()}
    assert terminal.values == expected_values
    assert emitted == expected_values
    assert json_values == expected_values
    assert csv_values == expected_values
    assert terminal.counts == expected_counts
    assert json_counts == expected_counts
    assert csv_counts == expected_counts
    assert "MUJOCO_WARNING" in terminal.text
    assert "WORKER_EXCEPTION" in terminal.text
    assert "CAPACITY_ANOMALY" in terminal.text
    assert "DANGEROUS_RATE_EXCEEDED" in terminal.text
    assert "CURRICULUM_STALLED" in terminal.text
    assert "constraint warning" in terminal.text


def test_native_tensorboard_writes_scalars_and_metadata(tmp_path: Path) -> None:
    # TensorBoard does not publish typing metadata.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore[import-untyped]

    snapshot = _aggregated_snapshot()
    metadata = ReportMetadata.from_config(load_run_config(DEFAULT_CONFIG_PATH))
    log_dir = tmp_path / "tensorboard"
    with TensorBoardTelemetryWriter(log_dir) as writer:
        writer.write_iteration(snapshot, metadata)
    accumulator = EventAccumulator(str(log_dir)).Reload()
    tag = TENSORBOARD_TAGS["throughput.steps_per_second"]
    assert accumulator.Scalars(tag)[0].value == pytest.approx(4096.0)
    assert accumulator.Scalars(f"{tag}/count")[0].value == 1.0
    assert "metadata/run" in accumulator.Tags()["tensors"]


def test_run_directory_manifest_tracks_real_artifacts(tmp_path: Path) -> None:
    snapshot = _aggregated_snapshot()
    config = load_run_config(DEFAULT_CONFIG_PATH)
    metadata = ReportMetadata.from_config(config)
    run_dir = prepare_run_directory(config, tmp_path / "run")
    incomplete = validate_run_directory(run_dir)
    assert not incomplete.complete
    assert "evaluation/final.json" in incomplete.missing
    assert "checkpoints/model_<iteration>.pt" in incomplete.missing

    write_final_reports(run_dir, snapshot, metadata)
    (run_dir / "checkpoints" / "model_3.pt").write_bytes(b"real producer fixture")
    (run_dir / "checkpoints" / "model_3.json").write_text("{}\n", encoding="utf-8")
    with TensorBoardTelemetryWriter(run_dir / "tensorboard") as writer:
        writer.write_iteration(snapshot, metadata)
    status = finalize_run_manifest(run_dir)
    assert status.complete
    status.require_complete()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["missing"] == []
    assert (run_dir / "config" / "resolved.yaml").is_file()
    assert (run_dir / "assets" / "manifest.json").is_file()


def test_multiple_diagnostics_are_never_filtered() -> None:
    aggregator = TelemetryAggregator()
    for kind, count in (
        (DiagnosticKind.MUJOCO, 2),
        (DiagnosticKind.MUJOCO, 3),
        (DiagnosticKind.SOLVER, 4),
        (DiagnosticKind.WORKER, 1),
    ):
        aggregator.record_diagnostic(DiagnosticEvent(0, TelemetryDiagnostic(kind, f"{kind.value} issue", count=count)))
    snapshot = aggregator.snapshot(0)
    assert snapshot.metrics["warning.mujoco.count"].value == 5.0
    assert snapshot.metrics["warning.solver.count"].value == 4.0
    assert len(snapshot.diagnostics) == 4
    summary = render_terminal_summary(snapshot)
    assert summary.text.count("DIAGNOSTIC[") == 4
