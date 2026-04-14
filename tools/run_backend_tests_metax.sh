#!/bin/bash

# Metax-specific backend test script
# Usage: bash tools/run_backend_tests_metax.sh metax

VENDOR=${1:?"Usage: bash tools/run_backend_tests_metax.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

source tools/run_command.sh

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

# Reduction ops
# FIXME(metax): error: invalid pointer element type: '!tt.ptr<i64>'
# run_command pytest -s tests/test_reduction_ops.py
# FIXME(metax): triton.compiler.errors.CompilationError
# run_command pytest -s tests/test_general_reduction_ops.py
# FIXME(metax): Killed
# run_command pytest -s tests/test_norm_ops.py

# Pointwise ops
### set private mem out of docker in case OOM
# modprobe -r metax
# modprobe metax pri_mem_sz=40
run_command pytest -s tests/test_pointwise_dynamic.py
run_command pytest -s tests/test_unary_pointwise_ops.py
# FIXME(metax): Fix AssertionError for this test
# run_command pytest -s tests/test_binary_pointwise_ops.py
run_command pytest -s tests/test_pointwise_type_promotion.py
run_command pytest -s tests/test_tensor_constructor_ops.py

# Blas ops
run_command pytest -s tests/test_blas_ops.py
# FIXME(metax): TypeError: 'list' object is not callable
# run_command pytest -s tests/test_attention_ops.py

# Special ops
# FIXME(metax): Fix core dump for this test
# run_command pytest -s tests/test_special_ops.py
run_command pytest -s tests/test_distribution_ops.py

# Utils
run_command pytest -s tests/test_libentry.py
run_command pytest -s tests/test_shape_utils.py
run_command pytest -s tests/test_tensor_wrapper.py

# Examples
# FIXME(metax): This test will fail for maybe bad internet connection
# run_command pytest -s examples/model_bert_test.py
