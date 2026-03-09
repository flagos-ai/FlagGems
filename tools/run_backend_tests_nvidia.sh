#!/bin/bash

# TODO(Qiming): Merge this to the generic run_backend_tests.sh file.
# TODO(Qiming): Make the test cases an input to the script

export CUDA_VISIBLE_DEVICES=6
# export http_proxy: ${{ secrets.HTTP_PROXY }}
# export https_proxy: ${{ secrets.HTTPS_PROXY }}

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

# TODO(Qiming): Remove the following conda activations
source "/home/zhangzhihui/miniconda3/etc/profile.d/conda.sh"
conda activate flag_gems

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
# run_command pytest -s tests/test_attention_ops.py
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
# TODO(Qiming): OSError: [Errno 101] Network is unreachable
# run_command pytest -s examples/model_bert_test.py
