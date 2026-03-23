#!/bin/bash

VENDOR=${1:?"Usage: bash tools/run_backend_tests_ascend.sh <vendor>"}
export GEMS_VENDOR=$VENDOR
export TRITON_ALL_BLOCKS_PARALLEL=1

# Initialize Ascend environment variables.
# This script is provided by the Huawei Ascend CANN toolkit installation.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

source tools/run_command.sh

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

# run_command python3 -m pytest -s tests/test_tensor_constructor_ops.py -k "not test_accuracy_randperm"
run_command python3 -m pytest -s tests/test_libentry.py
run_command python3 -m pytest -s tests/test_shape_utils.py
# run_command python3 -m pytest -s tests/test_tensor_wrapper.py
# run_command python3 -m pytest -s tests/test_distribution_ops.py
