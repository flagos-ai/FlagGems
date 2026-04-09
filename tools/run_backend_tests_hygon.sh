#!/bin/bash

VENDOR=${1:?"Usage: bash tools/run_backend_tests_hygon.sh <vendor>"}

# 1. Environment Activation
# Using pyenv to manage Python versions
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# 2. Dependency Installation

echo "Setting up Python environment..."
pip install -U pip
pip install uv

# Create virtual environment
uv venv
source .venv/bin/activate

# Install build tools
uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0

# Install project dependencies
# For Hygon, we typically install the ROCm version of PyTorch or the specific DTK version
# If you have a local .whl file (like the one you downloaded earlier), use:
# uv pip install /path/to/torch-2.x.x+das...whl
# Otherwise, try installing standard ROCm torch (requires ROCm drivers installed on system)
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# Install the project itself
uv pip install --no-build-isolation -e .

# Load Test Utilities
source tools/run_command.sh

echo "Starting tests..."

# 3. Execute Test Cases

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

echo "All tests finished."
