"""Run a short headless MuJoCo reset/step smoke check."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tienkung_rl._cli import StoreOnce, create_parser, parse_and_dispatch, positive_int


def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("smoke", __doc__ or "Run the headless smoke check.")
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=32,
        action=StoreOnce,
        help="Number of policy steps to execute headlessly.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return parse_and_dispatch(build_parser(), argv, runtime_module="tienkung_rl.validation.smoke")


if __name__ == "__main__":
    raise SystemExit(main())
