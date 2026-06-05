#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load MetaX backend environment variables
set -a
# shellcheck source=../flaggems.env
source "$SCRIPT_DIR/../flaggems.env"
set +a

# Run shared FlagGems installation logic
bash "$SCRIPT_DIR/../../common/scripts/install-flaggems.sh"
