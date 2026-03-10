#!/bin/bash

# Unified backend test script
# Usage: bash tools/run_backend_tests.sh <vendor>
# Example: bash tools/run_backend_tests.sh iluvatar

VENDOR=${1:?"Usage: bash tools/run_backend_tests.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

source tools/run_command.sh

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

DISPATCH_SCRIPT="tools/run_backend_tests_${VENDOR}.sh"
if [ -f "$DISPATCH_SCRIPT" ]; then
  bash "$DISPATCH_SCRIPT" "$VENDOR"
  exit $?
fi

echo "Unsupported backend vendor: $VENDOR (missing $DISPATCH_SCRIPT)"
exit 1
