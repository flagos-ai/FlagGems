#!/bin/bash

VENDOR=${1:-iluvatar}

#=== Common setup ===
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
pip install -U pip
pip install uv
uv venv
source .venv/bin/activate

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

#=== Install ===

export FLAGOS_PYPI=https://resource.flagos.net/repository/flagos-pypi-${VENDOR}/simple

uv pip install setuptools==79.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 'cmake>=3.20,<4.0' ninja==1.13.0
uv pip install --index $FLAGOS_PYPI \
       "torch==2.7.1+corex.4.4.0" "torchaudio==2.7.1+corex.4.4.0" \
       "torchvision==0.22.1+corex.4.4.0" \
       "triton==3.1.0+corex.4.4.0"
uv pip install -e .[iluvatar,test]

#=== Start testing ===

export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "Starting tests..."

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
    echo "Testing $testcase"
    pytest -s --tb=line $testcase --ref cpu
done
