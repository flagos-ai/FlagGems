#!/bin/bash

VENDOR=${1}
echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

export LD_LIBRARY_PATH=/xcudart/lib:/usr/local/cuda/lib64

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

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index https://download.pytorch.org/whl/cu118
uv pip install benchflow==1.0.0 torch_klx==0.1.0 torch_xray==0.2.1 triton==3.0.0+0762702f xmlir==1.0.0.1 \
  --index https://resource.flagos.net/repository/flagos-pypi-kunlunxin/simple

uv pip install -e .[kununxin,test]

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
