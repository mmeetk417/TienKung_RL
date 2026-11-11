"""Strict YAML loading, CLI overrides, validation and snapshot writing."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml  # type: ignore[import-untyped]
from yaml.constructor import ConstructorError  # type: ignore[import-untyped]

from tienkung_rl.paths import DEFAULT_CONFIG_PATH

from .models import ResolvedRunConfig

_ConfigT = TypeVar("_ConfigT")


class ConfigError(ValueError):
    """A configuration cannot be resolved without guessing its meaning."""


class DuplicateKeyError(ConfigError):
    """A YAML mapping repeats a key."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise DuplicateKeyError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}, "
                f"column {key_node.start_mark.column + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error
    try:
        value = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except DuplicateKeyError:
        raise
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a YAML mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError("configuration keys must be strings")
    return cast(dict[str, object], value)


def _parse_override_value(text: str, path: str) -> object:
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML override value for {path}: {error}") from error


def _normalise_overrides(
    overrides: Sequence[str] | Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    if overrides is None:
        return ()
    if isinstance(overrides, Mapping):
        return tuple((key, value) for key, value in overrides.items())
    if isinstance(overrides, (str, bytes)):
        overrides = (str(overrides),)
    result: list[tuple[str, object]] = []
    seen: set[str] = set()
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ConfigError("each override must use dotted.path=value syntax")
        path, raw_value = override.split("=", 1)
        path = path.strip()
        if path in seen:
            raise ConfigError(f"duplicate command-line override for {path!r}")
        seen.add(path)
        result.append((path, _parse_override_value(raw_value, path)))
    return tuple(result)


def apply_overrides(
    raw_config: Mapping[str, object],
    overrides: Sequence[str] | Mapping[str, object] | None,
) -> dict[str, object]:
    """Return a copied YAML tree with deterministic CLI-last overrides."""

    result = copy.deepcopy(dict(raw_config))
    for dotted_path, value in _normalise_overrides(overrides):
        if not isinstance(dotted_path, str):
            raise ConfigError("override paths must be strings")
        parts = dotted_path.split(".")
        if not dotted_path or any(not part for part in parts):
            raise ConfigError(f"invalid override path {dotted_path!r}")
        cursor: dict[str, object] = result
        for part in parts[:-1]:
            if part not in cursor:
                raise ConfigError(f"unknown override path {dotted_path!r} (missing {part!r})")
            child = cursor[part]
            if not isinstance(child, dict):
                raise ConfigError(f"override path {dotted_path!r} traverses non-mapping field {part!r}")
            cursor = child
        leaf = parts[-1]
        if leaf not in cursor:
            raise ConfigError(f"unknown override path {dotted_path!r}")
        cursor[leaf] = value
    return result


def _is_union(annotation: object) -> bool:
    return get_origin(annotation) in (Union, UnionType)


def _is_dataclass_type(annotation: object) -> TypeGuard[type[Any]]:
    return isinstance(annotation, type) and is_dataclass(annotation)


def _convert(value: object, annotation: object, path: str) -> object:
    if annotation is Any:
        raise ConfigError(f"{path}: untyped Any fields are forbidden")
    if _is_dataclass_type(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping for {annotation.__name__}, got {type(value).__name__}")
        return _construct_dataclass(annotation, value, path)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected YAML sequence, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(item, args[0], f"{path}[{index}]") for index, item in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} sequence items, got {len(value)}")
        return tuple(
            _convert(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, args, strict=True))
        )
    if _is_union(annotation):
        matches: list[object] = []
        errors: list[str] = []
        for item_type in args:
            try:
                matches.append(_convert(value, item_type, path))
            except ConfigError as error:
                errors.append(str(error))
        if len(matches) == 1:
            return matches[0]
        raise ConfigError(f"{path}: value does not match exactly one allowed type ({'; '.join(errors)})")
    if annotation is type(None) and value is None:
        return None
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigError(f"{path}: expected bool, got {type(value).__name__}")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigError(f"{path}: expected int, got {type(value).__name__}")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise ConfigError(f"{path}: expected float, got {type(value).__name__}")
        return float(cast(int | float, value))
    if annotation is str:
        if type(value) is not str:
            raise ConfigError(f"{path}: expected str, got {type(value).__name__}")
        return value
    raise ConfigError(f"{path}: unsupported configuration type {annotation!r}")


def _construct_dataclass(cls: type[_ConfigT], value: Mapping[str, object], path: str) -> _ConfigT:
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{path}: all field names must be strings")
    annotations = get_type_hints(cls)
    declared = {item.name: item for item in fields(cast(Any, cls))}
    unknown = sorted(set(value) - set(declared))
    if unknown:
        raise ConfigError(f"{path}: unknown field(s): {', '.join(unknown)}")
    missing = [
        name
        for name, item in declared.items()
        if name not in value and item.default is MISSING and item.default_factory is MISSING
    ]
    if missing:
        raise ConfigError(f"{path}: missing required field(s): {', '.join(missing)}")
    kwargs: dict[str, object] = {}
    for name in declared:
        if name in value:
            kwargs[name] = _convert(value[name], annotations[name], f"{path}.{name}")
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{path}: {error}") from error


def load_run_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: Sequence[str] | Mapping[str, object] | None = (),
) -> ResolvedRunConfig:
    """Load one YAML file, apply CLI-last overrides, and fully validate it."""

    config_path = Path(path).expanduser().resolve()
    raw = _load_yaml_mapping(config_path)
    overridden = apply_overrides(raw, overrides)
    config = _construct_dataclass(ResolvedRunConfig, overridden, "config")
    try:
        config.validate()
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return config


# Concise compatibility name for callers that already know the target type.
load_config = load_run_config


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(cast(Any, value))}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def resolved_yaml(config: ResolvedRunConfig) -> str:
    """Serialize a resolved config as deterministic, human-readable YAML."""

    config.validate()
    return yaml.safe_dump(
        _plain(config),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=100,
    )


def write_config_snapshot(
    config: ResolvedRunConfig,
    directory: str | Path,
    *,
    dependencies: Sequence[str] | None = None,
) -> str:
    """Write the resolved configuration and dependency snapshot files."""

    from .serialization import canonical_json, sha256_fingerprint

    config.validate()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    dependency_lines = (
        tuple(dependencies)
        if dependencies is not None
        else (
            f"mujoco=={config.platform.mujoco_version}",
            f"python=={config.platform.python_version}",
            f"torch=={config.platform.pytorch_version}",
        )
    )
    if any(not isinstance(item, str) or not item.strip() for item in dependency_lines):
        raise ConfigError("dependencies must contain non-empty strings")
    outputs = {
        "resolved.yaml": resolved_yaml(config),
        "canonical.json": canonical_json(config) + "\n",
        "dependencies.txt": "\n".join(sorted(set(dependency_lines))) + "\n",
    }
    for name, contents in outputs.items():
        (target / name).write_text(contents, encoding="utf-8", newline="\n")
    return sha256_fingerprint(config)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ConfigError",
    "DuplicateKeyError",
    "apply_overrides",
    "load_config",
    "load_run_config",
    "resolved_yaml",
    "write_config_snapshot",
]
