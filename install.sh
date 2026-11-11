#!/usr/bin/env bash
# Create the supported CPU environment, install this checkout, and verify it.

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT_NAME="${TIENKUNG_ENV_NAME:-tienkung-rl}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniconda or Miniforge as your user." >&2
  exit 2
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: the supported platform is Ubuntu 22.04 x86-64." >&2
  exit 2
fi
if [[ ! -r /etc/os-release ]] \
  || ! grep -Eq '^ID="?ubuntu"?$' /etc/os-release \
  || ! grep -Eq '^VERSION_ID="?22\.04"?$' /etc/os-release; then
  echo "ERROR: this locked installation supports Ubuntu 22.04 only." >&2
  exit 2
fi

conda env create \
  --name "${ENVIRONMENT_NAME}" \
  --file "${PROJECT_ROOT}/environment.lock.yml"

conda run --no-capture-output --name "${ENVIRONMENT_NAME}" \
  python -m pip install --no-deps --editable "${PROJECT_ROOT}"

conda run --no-capture-output --name "${ENVIRONMENT_NAME}" \
  python -m tienkung_rl.validation.installation

echo "Installation complete. Run: conda activate ${ENVIRONMENT_NAME}"
