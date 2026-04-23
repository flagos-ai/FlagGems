#!/bin/bash

# Unified backend test script
# Usage: bash tools/run_backend_tests.sh <vendor>
# Example: bash tools/run_backend_tests.sh iluvatar

VENDOR=${1:?"Usage: bash tools/run_backend_tests.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

source tools/run_command.sh

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

# Tensor constructor ops
run_command python3 -m pytest -s tests/test_tensor_constructor_ops.py

# Utils
run_command python3 -m pytest -s tests/test_libentry.py
run_command python3 -m pytest -s tests/test_shape_utils.py
run_command python3 -m pytest -s tests/test_tensor_wrapper.py

# Pointwise dynamic ops
run_command python3 -m pytest -s tests/test_pointwise_dynamic.py

# Distribution ops
run_command python3 -m pytest -s tests/test_distribution_ops.py

# FIXME(moore): Softmax only support float32/float16/bfloat16
# run_command python3 -m pytest -s tests/test_reduction_ops.py
# FIXME(moore): BatchNorm supports Float/Half/BFloat16 input dtype
# run_command python3 -m pytest -s tests/test_norm_ops.py
# FIXME(moore): RuntimeError: _Map_base::at (missing operators)
# run_command python3 -m pytest -s tests/test_unary_pointwise_ops.py
# FIXME(moore): unsupported data type DOUBLE
# run_command python3 -m pytest -s tests/test_blas_ops.py
