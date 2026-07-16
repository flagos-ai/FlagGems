#!/usr/bin/env bash
# Called by devcontainer.json postCreateCommand.
# FLAGGEMS_DEVCONTAINER_BACKEND is set via containerEnv in devcontainer.json.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/../../common/scripts/install-flaggems.sh"
