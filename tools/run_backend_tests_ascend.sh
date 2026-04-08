#!/bin/bash

VENDOR=${1}
export GEMS_VENDOR=$VENDOR

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

# TODO: Check if this is necessary
export TRITON_ALL_BLOCKS_PARALLEL=1

# Initialize Ascend environment variables.
# This script is provided by the Huawei Ascend CANN toolkit installation.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Set virtual environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

pip install -U pip
pip install uv
uv venv
source .venv/bin/activate

uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0
uv pip install flagtree==0.5.0+3.2 torch==2.9.0+cpu torch-npu==2.9.0 \
  -i https://resource.flagos.net/repository/flagos-pypi-ascend/simple
uv pip install -e .[ascend,test]

# Start testing
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
    pytest -s --tb=line $testcase
done
