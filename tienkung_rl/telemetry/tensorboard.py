"""TensorBoard output for stable telemetry snapshots.

Scalar values are written only through :meth:`write_iteration`, making the
iteration boundary explicit.  The native writer uses TensorBoard's protobuf
event writer directly and therefore does not require importing Torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from tienkung_rl.config.serialization import canonical_json

from .metrics import METRIC_SCHEMA, TelemetrySnapshot

TENSORBOARD_TAGS: Mapping[str, str] = {
    name: f"{spec.level.value}/{name.replace('.', '/')}" for name, spec in METRIC_SCHEMA.items()
}
if len(set(TENSORBOARD_TAGS.values())) != len(TENSORBOARD_TAGS):  # pragma: no cover
    raise RuntimeError("TensorBoard tag mapping contains a collision")


@runtime_checkable
class SummaryWriterLike(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> object: ...

    def add_text(self, tag: str, text_string: str, global_step: int) -> object: ...

    def flush(self) -> object: ...

    def close(self) -> object: ...


class NativeSummaryWriter:
    """Small Torch-independent adapter over TensorBoard's event-file writer."""

    def __init__(self, log_dir: str | Path) -> None:
        # TensorBoard does not publish typing metadata.
        from tensorboard.summary.writer.event_file_writer import EventFileWriter  # type: ignore[import-untyped]

        target = Path(log_dir)
        target.mkdir(parents=True, exist_ok=True)
        self._writer = EventFileWriter(str(target))

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        import time

        from tensorboard.compat.proto import (  # type: ignore[import-untyped]  # Generated protobuf modules have no typing metadata.
            event_pb2,
            summary_pb2,
        )

        summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(scalar_value))])
        self._writer.add_event(event_pb2.Event(wall_time=time.time(), step=global_step, summary=summary))

    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        import time

        from tensorboard.compat.proto import (
            event_pb2,
            summary_pb2,
            tensor_pb2,
            tensor_shape_pb2,
            types_pb2,
        )

        metadata = summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="text"))
        tensor = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING,
            string_val=[text_string.encode("utf-8")],
            tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]),
        )
        summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, metadata=metadata, tensor=tensor)])
        self._writer.add_event(event_pb2.Event(wall_time=time.time(), step=global_step, summary=summary))

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def _metadata_dict(metadata: Mapping[str, object] | object) -> Mapping[str, object]:
    if isinstance(metadata, Mapping):
        result = metadata
    else:
        converter = getattr(metadata, "to_dict", None)
        if not callable(converter):
            raise TypeError("TensorBoard metadata must be a mapping or expose to_dict()")
        result = converter()
        if not isinstance(result, Mapping):
            raise TypeError("metadata to_dict() must return a mapping")
    required = {"seed", "command", "config_fingerprint", "asset_fingerprints"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"TensorBoard metadata is missing {sorted(missing)}")
    return result


class TensorBoardTelemetryWriter:
    """Write every schema metric and its sample count at iteration boundaries."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        *,
        writer: SummaryWriterLike | None = None,
    ) -> None:
        if (log_dir is None) == (writer is None):
            raise ValueError("provide exactly one of log_dir or writer")
        if writer is None:
            assert log_dir is not None
            self._writer: SummaryWriterLike = NativeSummaryWriter(log_dir)
        else:
            self._writer = writer
        self._last_iteration: int | None = None
        self._metadata_json: str | None = None

    def write_iteration(
        self,
        snapshot: TelemetrySnapshot,
        metadata: Mapping[str, object] | object,
    ) -> Mapping[str, float]:
        if self._last_iteration is not None and snapshot.iteration <= self._last_iteration:
            raise ValueError("TensorBoard iterations must be strictly increasing")
        metadata_json = canonical_json(_metadata_dict(metadata))
        if self._metadata_json is None:
            self._writer.add_text("metadata/run", metadata_json, snapshot.iteration)
            self._metadata_json = metadata_json
        elif self._metadata_json != metadata_json:
            raise ValueError("run metadata cannot change between TensorBoard iterations")

        emitted: dict[str, float] = {}
        for name in METRIC_SCHEMA:
            aggregate = snapshot.metrics[name]
            tag = TENSORBOARD_TAGS[name]
            self._writer.add_scalar(tag, aggregate.value, snapshot.iteration)
            self._writer.add_scalar(f"{tag}/count", float(aggregate.count), snapshot.iteration)
            emitted[name] = aggregate.value
        self._writer.flush()
        self._last_iteration = snapshot.iteration
        return emitted

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> TensorBoardTelemetryWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "TENSORBOARD_TAGS",
    "NativeSummaryWriter",
    "SummaryWriterLike",
    "TensorBoardTelemetryWriter",
]
