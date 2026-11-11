"""Collect the bounded M05 native-MuJoCo parallel baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# These imports follow the direct-script repository bootstrap above.
from tienkung_rl.config import DEFAULT_CONFIG_PATH, load_run_config  # noqa: E402
from tienkung_rl.validation.parallel_profile import (  # noqa: E402
    profile_parallel_executor,
    write_parallel_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--env-counts", type=int, nargs="+", default=(1, 4, 128))
    parser.add_argument(
        "--worker-counts",
        type=int,
        nargs="+",
        default=None,
        help="Worker-thread counts to cross with each compatible environment count.",
    )
    parser.add_argument("--warmup-policy-steps", type=int, default=1)
    parser.add_argument("--measured-policy-steps", type=int, default=3)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/profiling/m05_parallel_baseline.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/profiling/m05_parallel_baseline.md"),
    )
    args = parser.parse_args()
    config = load_run_config(args.config)
    report = profile_parallel_executor(
        config,
        environment_counts=args.env_counts,
        worker_thread_counts=args.worker_counts,
        warmup_policy_steps=args.warmup_policy_steps,
        measured_policy_steps=args.measured_policy_steps,
    )
    write_parallel_profile(
        report,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
    )
    print(f"wrote {args.output_json} and {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
