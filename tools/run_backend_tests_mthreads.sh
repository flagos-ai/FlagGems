#!/bin/bash

VENDOR=${1}
echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

# PyEnv settings
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Preamble
pip install -U pip
pip install uv
uv venv
source .venv/bin/activate

# Setup
uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0
uv pip install -e .[methreads,test]

# For the intel math library
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH

# In case the backend detection fails
# export GEMS_VENDOR=$VENDOR
source tools/run_command.sh

# Reduction ops
# FIXME(moore): Softmax only support float32/float16/bfloat16
pytest -s tests/test_reduction_ops.py
pytest -s tests/test_general_reduction_ops.py
# FIXME(moore): BatchNorm supports Float/Half/BFloat16 input dtype
pytest -s tests/test_norm_ops.py

# Pointwise ops
pytest -s tests/test_pointwise_dynamic.py
# FIXME(moore): RuntimeError: _Map_base::at (missing operators)
pytest -s tests/test_unary_pointwise_ops.py
pytest -s tests/test_binary_pointwise_ops.py
pytest -s tests/test_pointwise_type_promotion.py

# TODO: test_accuracy_randperm
pytest -s tests/test_tensor_constructor_ops.py

# BLAS ops
# TODO(Qiming): Fix sharedencoding on Hopper
pytest -s tests/test_attention_ops.py
# FIXME(moore): unsupported data type DOUBLE
pytest -s tests/test_blas_ops.py

# Special ops
pytest -s tests/test_special_ops.py

# Distribution
pytest -s tests/test_distribution_ops.py

# Convolution ops
pytest -s tests/test_convolution_ops.py

# Utils
pytest -s tests/test_libentry.py
pytest -s tests/test_shape_utils.py
pytest -s tests/test_tensor_wrapper.py

# Examples
# TODO(Qiming): OSError: [Errno 101] Network is unreachable
# run_command pytest -s examples/model_bert_test.py
