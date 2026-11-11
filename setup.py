# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from pathlib import Path

from setuptools import find_packages, setup

TIENKUNG_PACKAGES = find_packages(
    where=".",
    include=("tienkung_rl", "tienkung_rl.*"),
)
RSL_RL_PACKAGES = find_packages(
    where="rsl_rl",
    include=("rsl_rl", "rsl_rl.*"),
)
ASSET_ROOT = Path("tienkung_rl/assets/tienkung2_pro")
ASSET_PACKAGE_DATA = [
    path.relative_to("tienkung_rl").as_posix() for path in sorted(ASSET_ROOT.rglob("*")) if path.is_file()
]
FROZEN_MANIFEST_ROOT = Path("tienkung_rl/learning/manifest_data")
ASSET_PACKAGE_DATA.extend(
    path.relative_to("tienkung_rl").as_posix() for path in sorted(FROZEN_MANIFEST_ROOT.glob("*.json")) if path.is_file()
)

setup(
    name="tienkung-reach-rl",
    packages=TIENKUNG_PACKAGES + RSL_RL_PACKAGES,
    package_dir={"rsl_rl": "rsl_rl/rsl_rl"},
    package_data={"tienkung_rl": ASSET_PACKAGE_DATA},
    version="0.1.0",
    description="MuJoCo CPU reach-then-close training for TienKung 2 Pro",
    python_requires=">=3.11,<3.12",
    install_requires=[
        "mujoco==3.11.0",
        "numpy==2.3.2",
        "PyYAML==6.0.2",
        "tensorboard==2.20.0",
        "torch==2.13.0",
    ],
    include_package_data=True,
)
