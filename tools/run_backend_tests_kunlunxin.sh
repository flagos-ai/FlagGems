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
uv pip install setuptools==79.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0

uv pip install \
    nvidia-cublas-cu11==11.11.3.6 \
    nvidia-cuda-cupti-cu11==11.8.87 \
    nvidia-cuda-nvrtc-cu11==11.8.89 \
    nvidia-cuda-runtime-cu11==11.8.89 \
    nvidia-cudnn-cu11==9.1.0.70 \
    nvidia-cufft-cu11==10.9.0.58 \
    nvidia-curand-cu11==10.3.0.86 \
    nvidia-cusolver-cu11==11.4.1.48 \
    nvidia-cusparse-cu11==11.7.5.86 \
    nvidia-nccl-cu11==2.21.5 \
    nvidia-nvtx-cu11==11.8.86 \
  --index https://resource.flagos.net/respository/flagos-pypi-kunlunxin/simple

uv pip install -e .[kunlunxin,test]

$HOME/kunlunxin/install-wheels.sh

uv pip uninstall pytest-repeat pytest-timeout

uv pip list

echo "Start running tests ..."

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
