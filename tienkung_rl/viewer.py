"""View one MuJoCo environment with task diagnostics."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from tienkung_rl._cli import StoreOnce, create_parser, dispatch


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _brightness(text: str) -> float:
    value = _finite_float(text)
    if value < 0.0 or value > 10.0:
        raise argparse.ArgumentTypeError("must be between 0 and 10")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = create_parser("viewer", __doc__ or "Open the viewer.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        action=StoreOnce,
        help="New MuJoCo checkpoint for deterministic inference.",
    )
    parser.add_argument(
        "--mode",
        choices=("policy", "random", "zero", "scripted"),
        default="zero",
        action=StoreOnce,
        help="Controller used in the single displayed environment.",
    )
    parser.add_argument(
        "--target-x-m", type=_finite_float, action=StoreOnce, help="Forced cylinder/target world X (m)."
    )
    parser.add_argument(
        "--target-y-m", type=_finite_float, action=StoreOnce, help="Forced cylinder/target world Y (m)."
    )
    parser.add_argument("--cylinder-radius-m", type=_finite_float, action=StoreOnce, help="Forced cylinder radius (m).")
    parser.add_argument("--cylinder-height-m", type=_finite_float, action=StoreOnce, help="Forced cylinder height (m).")
    parser.add_argument("--cylinder-mass-kg", type=_finite_float, action=StoreOnce, help="Forced cylinder mass (kg).")
    parser.add_argument(
        "--hand-sliding-friction",
        type=_finite_float,
        action=StoreOnce,
        help="Forced hand/cylinder sliding friction.",
    )
    parser.add_argument(
        "--table-sliding-friction",
        type=_finite_float,
        action=StoreOnce,
        help="Forced table/cylinder sliding friction.",
    )
    collision_visibility = parser.add_mutually_exclusive_group()
    collision_visibility.add_argument(
        "--show-collisions",
        action="store_true",
        help="Show simplified robot collision geometry (hidden by default).",
    )
    collision_visibility.add_argument(
        "--hide-collisions",
        action="store_false",
        dest="show_collisions",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(show_collisions=False)
    parser.add_argument(
        "--hide-debug-geometry",
        action="store_true",
        help="Hide target, admission, contact-point, normal, and force debug geometry.",
    )
    parser.add_argument(
        "--hide-overlays",
        action="store_true",
        help="Hide the structured diagnostic text overlay.",
    )
    parser.add_argument(
        "--pause-on-terminal",
        action="store_true",
        help="Hold the terminal pose instead of immediately resetting the displayed episode.",
    )
    parser.add_argument(
        "--brightness",
        type=_brightness,
        default=1.5,
        action=StoreOnce,
        help="Scale native viewer headlight brightness (0=dark, 1=default, 10=max).",
    )
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=None,
        action=StoreOnce,
        help="Automation probe: run this many policy steps without creating a GUI.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "policy" and args.checkpoint is None:
        parser.error("--mode policy requires --checkpoint")
    if args.mode != "policy" and args.checkpoint is not None:
        parser.error("--checkpoint may only be used with --mode policy")
    if (args.target_x_m is None) != (args.target_y_m is None):
        parser.error("--target-x-m and --target-y-m must be supplied together")
    if args.headless_steps is not None and args.headless_steps < 1:
        parser.error("--headless-steps must be positive")
    return dispatch(parser, args, runtime_module="tienkung_rl.visualization.viewer")


if __name__ == "__main__":
    raise SystemExit(main())
