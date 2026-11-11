"""Shared, dependency-light command-line parsing for all public commands."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


class CliUsageError(ValueError):
    """An error that should be rendered as an argparse usage error."""


class StoreOnce(argparse.Action):
    """Reject duplicate scalar options instead of silently taking the last."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        seen: set[str] = getattr(namespace, "_explicit_options", set())
        key = option_string or self.dest
        if key in seen:
            parser.error(f"{key} may be specified only once")
        namespace._explicit_options = {*seen, key}
        setattr(namespace, self.dest, values)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared config, reproducibility, output, and CPU rollout flags."""

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        action=StoreOnce,
        help=("YAML run configuration (default: configs/reach_then_close.yaml)."),
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=None,
        action=StoreOnce,
        help="Random seed as a non-negative integer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        action=StoreOnce,
        help="Directory for this run's checkpoints, logs, and reports.",
    )
    parser.add_argument(
        "--num-envs",
        type=positive_int,
        default=None,
        action=StoreOnce,
        help="Number of independent CPU simulation environments.",
    )
    parser.add_argument(
        "--worker-threads",
        type=positive_int,
        default=None,
        action=StoreOnce,
        help="Number of in-process MuJoCo CPU worker threads.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help=("Override one typed YAML field; repeat for multiple distinct fields."),
    )


def create_parser(command: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m tienkung_rl.{command}",
        description=description,
    )
    add_common_arguments(parser)
    return parser


_COMMON_CONFIG_PATHS = {
    "seed": "run.seed",
    "output_dir": "run.output_dir",
    "num_envs": "platform.num_envs",
    "worker_threads": "platform.worker_threads",
}


def _encoded_override(path: str, value: object) -> str:
    if isinstance(value, Path):
        value = str(value)
    return f"{path}={json.dumps(value, ensure_ascii=False)}"


def config_overrides(args: argparse.Namespace) -> tuple[str, ...]:
    """Return validated config overrides from common and ``--set`` options."""

    overrides: list[str] = []
    paths: set[str] = set()
    for item in args.config_overrides:
        if "=" not in item:
            raise CliUsageError(f"invalid --set {item!r}; expected SECTION.FIELD=VALUE")
        path, _, _ = item.partition("=")
        if not path or path.strip() != path:
            raise CliUsageError(f"invalid --set field path {path!r}")
        if path in paths:
            raise CliUsageError(f"configuration field {path!r} is overridden twice")
        paths.add(path)
        overrides.append(item)

    for destination, path in _COMMON_CONFIG_PATHS.items():
        value = getattr(args, destination)
        if value is None:
            continue
        if path in paths:
            raise CliUsageError(f"{path!r} is set by both --{destination.replace('_', '-')} and --set")
        paths.add(path)
        overrides.append(_encoded_override(path, value))
    return tuple(overrides)


def load_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Any:
    """Load M01's strict configuration and render all failures consistently."""

    try:
        overrides = config_overrides(args)
        from tienkung_rl.config.load import DEFAULT_CONFIG_PATH, load_run_config

        config_path = DEFAULT_CONFIG_PATH if args.config is None else args.config
        return load_run_config(config_path, overrides=overrides)
    except (CliUsageError, FileNotFoundError, TypeError, ValueError) as error:
        parser.error(f"configuration error: {error}")


def dispatch(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    runtime_module: str,
    runtime_function: str = "run",
) -> int:
    """Load config, then lazily call a command runtime supplied by later modules."""

    config = load_config(parser, args)
    try:
        module = importlib.import_module(runtime_module)
        runtime: Callable[[Any, argparse.Namespace], int | None] = getattr(module, runtime_function)
    except (ImportError, AttributeError) as error:
        parser.error(f"runtime integration {runtime_module}.{runtime_function} is unavailable: {error}")
    result = runtime(config, args)
    return 0 if result is None else int(result)


def parse_and_dispatch(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    *,
    runtime_module: str,
    runtime_function: str = "run",
) -> int:
    args = parser.parse_args(argv)
    return dispatch(
        parser,
        args,
        runtime_module=runtime_module,
        runtime_function=runtime_function,
    )
