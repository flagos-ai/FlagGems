#!/bin/bash

VENDOR=${1}
echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

# Tsingmicro dependencies
export TX8_DEPS_ROOT=/opt/tsingmicro/tx8_deps
export LLVM_SYSPATH=/opt/tsingmicro/llvm21
export LLVM_BINARY_DIR=${LLVM_SYSPATH}/bin
export PYTHONPATH=${LLVM_SYSPATH}/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:/usr/local/kuiper/lib:$LD_LIBRARY_PATH
export FLAGOS_PYPI=https://resource.flagos.net/repository/flagos-pypi-tsingmicro/simple

# PyEnv settings
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

uv venv
source .venv/bin/activate

# Setup
uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0
uv pip install torch_txda==0.1.0+20260310.294fc4a6 --index $FLAGOS_PYPI
uv pip uninstall triton
# Use flagtree
uv pip install flagtree==0.5.0+tsingmicro3.3 --index $FLAGOS_PYPI
# uv pip install triton==3.3.0+gitfe2a28fa --index $FLAGOS_PYPI
uv pip install txops==0.1.0+20260225.5cc33e4e --index $FLAGOS_PYPI

uv pip install -e .[tsingmicro,test]

# The following is needed when using `triton==3.3.0+gitfe2a28fa` rather than `flagtree`
# SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.10/site-packages
# export PYTHONPATH=$SITE_PACKAGES/triton/backends/tsingmicro/llvm/python_packages/mlir_core

# NOTE: The following setting may be needed if there are exceptions related to txops.
# export LD_LIBRARY_PATH=$SITE_PACKAGES/txops/lib:$LD_LIBRARY_PATH

# In case the backend detection fails
# export GEMS_VENDOR=$VENDOR

TEST_FILES=(
  # Reduction
  "tests/test_reduction_ops.py"
  "tests/test_general_reduction_ops.py"
  "tests/test_norm_ops.py"
  # Pointwise
  "tests/test_pointwise_dynamic.py"
  "tests/test_unary_pointwise_ops.py"
  "tests/test_binary_pointwise_ops.py"
  "tests/test_pointwise_type_promotion.py"
  # Tensor
  "tests/test_tensor_constructor_ops.py"
  "tests/test_tensor_wrapper.py"
  # Attention
  "tests/test_attention_ops.py"
  "tests/test_blas_ops.py"
  # Special
  "tests/test_special_ops.py"
  # Distribution
  "tests/test_distribution_ops.py"
  # Convolution
  "tests/test_convolution_ops.py"
  # Utils
  "tests/test_libentry"
  "tests/test_shape_utils.py"
  # DSA
  "tests/test_DSA/test_bin_topk.py"
  "tests/test_DSA/test_sparse_mla_ops.py"
  "tests/test_DSA/test_indexer_k_tiled.py"
  # FLA
  "tests/test_FLA/test_fla_utils_input_guard.py"
  "tests/test_FLA/test_fused_recurrent_gated_delta_rule.py"
)

for testcase in "${TEST_FILES[@]}"; do
    pytest -s --tb=line $testcase --ref cpu
done
