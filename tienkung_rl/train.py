"""Train the MuJoCo CPU reach-then-close policy from scratch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tienkung_rl._cli import create_parser, parse_and_dispatch, positive_int


def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("train", __doc__ or "Train a policy.")
    parser.add_argument(
        "--max-iterations",
        type=positive_int,
        default=None,
        help="Maximum number of PPO learning iterations.",
    )
    parser.add_argument(
        "--save-interval",
        type=positive_int,
        default=None,
        help="Save a checkpoint every this many learning iterations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the new MuJoCo checkpoint selected by configuration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return parse_and_dispatch(build_parser(), argv, runtime_module="tienkung_rl.learning.runner")


if __name__ == "__main__":
    raise SystemExit(main())
