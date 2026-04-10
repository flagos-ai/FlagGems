#!/bin/bash

VENDOR=${1:?"Usage: bash tools/run_backend_tests_iluvatar.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

pip install -U pip
pip install uv
uv venv
source .venv/bin/activate
uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0
uv pip install -e .[iluvatar,test]

source tools/run_command.sh

# Reduction ops
run_command pytest -s tests/test_reduction_ops.py
run_command pytest -s tests/test_general_reduction_ops.py
run_command pytest -s tests/test_norm_ops.py

# Pointwise ops
run_command pytest -s tests/test_pointwise_dynamic.py
run_command pytest -s tests/test_unary_pointwise_ops.py
run_command pytest -s tests/test_binary_pointwise_ops.py
run_command pytest -s tests/test_pointwise_type_promotion.py
run_command pytest -s tests/test_tensor_constructor_ops.py

# BLAS ops
# TODO(Qiming): Fix sharedencoding on Hopper
run_command pytest -s tests/test_attention_ops.py
run_command pytest -s tests/test_blas_ops.py

# Special ops
run_command pytest -s tests/test_special_ops.py
run_command pytest -s tests/test_distribution_ops.py

# Convolution ops
run_command pytest -s tests/test_convolution_ops.py

# Utils
run_command pytest -s tests/test_libentry.py
run_command pytest -s tests/test_shape_utils.py
run_command pytest -s tests/test_tensor_wrapper.py
