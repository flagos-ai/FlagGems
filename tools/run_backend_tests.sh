#!/bin/bash

# Unified backend test script
# Usage: bash tools/run_backend_tests.sh <vendor>
# Example: bash tools/run_backend_tests.sh iluvatar

VENDOR=${1:?"Usage: bash tools/run_backend_tests.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

source tools/run_command.sh

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

if [ "$VENDOR" == "iluvatar" ]; then
  export PATH=/usr/local/corex/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
fi

# Setup
sudo apt update
# NOTE: Iluvatar runner has ubuntu 20.04, with Python 3.8 as default
sudo apt install python3 python3-dev python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake uv
uv pip install -e '.[test]'

# Reduction ops
if [ "$VENDOR" != "moore" ]; then
  # FIXME(moore): Softmax only support float32/float16/bfloat16
  run_command python3 -m pytest -s tests/test_reduction_ops.py
  # FIXME(moore): BatchNorm supports Float/Half/BFloat16 input dtype
  run_command python3 -m pytest -s tests/test_norm_ops.py
fi
run_command python3 -m pytest -s tests/test_general_reduction_ops.py

# Pointwise ops
run_command python3 -m pytest -s tests/test_pointwise_dynamic.py
if [ "$VENDOR" != "moore" ]; then
  # FIXME(moore): RuntimeError: _Map_base::at (missing operators)
  run_command python3 -m pytest -s tests/test_unary_pointwise_ops.py
fi
run_command python3 -m pytest -s tests/test_binary_pointwise_ops.py
run_command python3 -m pytest -s tests/test_pointwise_type_promotion.py
run_command python3 -m pytest -s tests/test_tensor_constructor_ops.py

# BLAS ops
run_command python3 -m pytest -s tests/test_attention_ops.py
if [ "$VENDOR" != "moore" ]; then
  # FIXME(moore): unsupported data type DOUBLE
  run_command python3 -m pytest -s tests/test_blas_ops.py
fi

# Special ops
run_command python3 -m pytest -s tests/test_special_ops.py
run_command python3 -m pytest -s tests/test_distribution_ops.py

# Convolution ops
run_command python3 -m pytest -s tests/test_convolution_ops.py

# Utils
run_command python3 -m pytest -s tests/test_libentry.py
run_command python3 -m pytest -s tests/test_shape_utils.py
run_command python3 -m pytest -s tests/test_tensor_wrapper.py
