#!/bin/bash

VENDOR=${1:?"Usage: bash tools/run_backend_tests_thead.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

# 2. Device Configuration
# T-Head PPUs typically use the PPU_VISIBLE_DEVICES environment variable
# Based on previous logs, device IDs are 0, 1, 2, 3. Defaulting to device 0 for testing.
# export PPU_VISIBLE_DEVICES=0

# echo "Running FlagGems tests on T-Head PPU with GEMS_VENDOR=$GEMS_VENDOR"
# echo "Target Device: PPU $PPU_VISIBLE_DEVICES"

# 3. Environment Activation
# Using pyenv to manage Python versions
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# 4. Dependency Installation

echo "Setting up Python environment..."
pip install -U pip
pip install uv

# Create virtual environment
uv venv
source .venv/bin/activate

# Install build tools
uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0

# Install project dependencies
# If T-Head has specific extra dependency groups (e.g., .[tpu] or .[custom]), please modify here.
uv pip install --no-build-isolation -e .

# If specific T-Head runtime libraries are required (e.g., torch-ppu), add them here
# uv pip install torch-ppu ...

# 5. Load Test Utilities
source tools/run_command.sh

echo "Starting tests..."

# 6. Execute Test Cases

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
# Note: If T-Head does not support certain specific Attention mechanisms, this test may need to be skipped
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

# Examples
# Kept commented out as the network may be unreachable
# run_command pytest -s examples/model_bert_test.py

echo "All tests finished."
