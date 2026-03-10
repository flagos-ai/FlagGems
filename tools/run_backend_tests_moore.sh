#!/bin/bash

VENDOR=${1:?"Usage: bash tools/run_backend_tests_moore.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

source tools/run_command.sh

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

run_command python3 -m pytest -s tests/test_tensor_constructor_ops.py -k "not test_accuracy_randperm"
run_command python3 -m pytest -s tests/test_libentry.py
run_command python3 -m pytest -s tests/test_shape_utils.py
run_command python3 -m pytest -s tests/test_tensor_wrapper.py
run_command python3 -m pytest -s tests/test_pointwise_dynamic.py
run_command python3 -m pytest -s tests/test_distribution_ops.py
