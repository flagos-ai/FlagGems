#!/bin/bash

VENDOR=${1}
echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

# TODO(Qiming): This is a temporary hack. The container has a working library
# while the host doesn't. This one is copied from the container instance.
export KUIPER_HOME=/home/secure/runtime/kuiper
export PATH=$KUIPER_HOME/bin:$PATH
export LD_LIBRARY_PATH=$KUIPER_HOME/lib:$LD_LIBRARY_PATH

# PyEnv settings
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

uv venv
source .venv/bin/activate

# Setup
uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0
uv pip install torch_txda==0.1.0+20260310.294fc4a6 --index https://resource.flagos.net/repository/flagos-pypi-tsingmicro/simple
uv pip install triton==3.3.0+gitfe2a28fa --index https://resource.flagos.net/repository/flagos-pypi-tsingmicro/simple
uv pip install txops==0.1.0+20260225.5cc33e4e --index https://resource.flagos.net/repository/flagos-pypi-tsingmicro/simple

uv pip install -e .[tsingmicro,test]

# For the Triton library
SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$SITE_PACKAGES/txops/lib:$LD_LIBRARY_PATH
# NOTE: special settings for triton==3.3.0+gitfe2a28fa
export PYTHONPATH=$SITE_PACKAGES/triton/backends/tsingmicro/llvm/python_packages/mlir_core

# In case the backend detection fails
# export GEMS_VENDOR=$VENDOR
TEST_CMD="pytest -s --tb=line"
TEST_MODE="--ref cpu"

# Reduction ops
$CMD tests/test_reduction_ops.py $TEST_MODE
$TEST_CMD tests/test_general_reduction_ops.py $TEST_MODE
$TEST_CMD tests/test_norm_ops.py $TEST_MODE

# Pointwise ops
$TEST_CMD tests/test_pointwise_dynamic.py $TEST_MODE
$TEST_CMD tests/test_unary_pointwise_ops.py $TEST_MODE
$TEST_CMD tests/test_binary_pointwise_ops.py $TEST_MODE
$TEST_CMD tests/test_pointwise_type_promotion.py $TEST_MODE

$TEST_CMD tests/test_tensor_constructor_ops.py $TEST_MODE

# BLAS ops
$TEST_CMD tests/test_attention_ops.py $TEST_MODE
$TEST_CMD tests/test_blas_ops.py $TEST_MODE

# Special ops
$TEST_CMD tests/test_special_ops.py $TEST_MODE

# Distribution
$TEST_CMD tests/test_distribution_ops.py $TEST_MODE

# Convolution ops
$TEST_CMD tests/test_convolution_ops.py $TEST_MODE

# Utils
$TEST_CMD tests/test_libentry.py $TEST_MODE
$TEST_CMD tests/test_shape_utils.py $TEST_MODE
$TEST_CMD tests/test_tensor_wrapper.py $TEST_MODE
