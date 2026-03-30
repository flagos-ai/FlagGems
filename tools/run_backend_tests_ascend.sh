#!/bin/bash

# For testing
CHANGED_FILES="__ALL__"

VENDOR=${1:?"Usage: bash tools/run_backend_tests_ascend.sh <vendor>"}
export GEMS_VENDOR=$VENDOR
export TRITON_ALL_BLOCKS_PARALLEL=1

if [[ "$CHANGED_FILES" == "__ALL__" ]]; then
  # Replace "__ALL__" with all tests
  CHANGED_FILES=$(find tests -name "test*.py")
  EXTRA_OPTS=""
else
  # for per-PR test, fail early
  EXTRA_OPTS="-x"
fi

TEST_CASES=()
for item in $CHANGED_FILES; do
  case $item in
    # tests/test_DSA/*)
    # skip DSA test for now
    #  ;;
    # tests/test_quant.py)
    # skip because it always fail
    #  ;;
    tests/*) TEST_CASES+=($item)
  esac
done

# Skip tests if no tests file is found
if [ ${#TEST_CASES[@]} -eq 0 ]; then
  exit 0
fi

# Initialize Ascend environment variables.
# This script is provided by the Huawei Ascend CANN toolkit installation.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

source tools/run_command.sh

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

python3 -m pytest -s ${EXTRA_OPTS} ${TEST_CASES[@]}
