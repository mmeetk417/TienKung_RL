"""Validate the generated MuJoCo robot and scene assets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tienkung_rl._cli import StoreOnce, create_parser, parse_and_dispatch


def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("validate_asset", __doc__ or "Validate assets.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        action=StoreOnce,
        help="Optional JSON validation report path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return parse_and_dispatch(build_parser(), argv, runtime_module="tienkung_rl.assets.validation")


if __name__ == "__main__":
    raise SystemExit(main())
