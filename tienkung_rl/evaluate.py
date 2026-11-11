"""Evaluate a MuJoCo checkpoint with a deterministic seven-joint policy."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tienkung_rl._cli import StoreOnce, create_parser, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("evaluate", __doc__ or "Evaluate a policy.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        action=StoreOnce,
        help="MuJoCo seven-joint policy checkpoint to evaluate.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        action=StoreOnce,
        help="Frozen validation or final manifest; every listed case is replayed exactly once.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(parser, args)
    if args.checkpoint is None or args.manifest is None:
        parser.error("evaluation requires explicit --checkpoint and --manifest")
    from tienkung_rl.learning.evaluation import run

    return run(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
