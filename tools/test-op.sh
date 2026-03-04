#!/bin/bash

set -e

# Test cases that needs to run quick cpu tests
QUICK_CPU_TESTS=(
  "tests/test_attention_ops.py"
  "tests/test_binary_pointwise_ops.py"
  "tests/test_blas_ops.py"
  "tests/test_general_reduction_ops.py"
  "tests/test_norm_ops.py"
  "tests/test_pointwise_type_promotion.py"
  "tests/test_reduction_ops.py"
  "tests/test_special_ops.py"
  "tests/test_tensor_constructor_ops.py"
  "tests/test_unary_pointwise_ops.py"
)

PR_ID=$1
ID_SHA="${PR_ID}-${GITHUB_SHA::7}"
echo "CHANGED_FILES=$CHANGED_FILES"

TEST_CASES=()
TEST_CASES_CPU=()
UNARY_POINTWISE=0
for item in $CHANGED_FILES; do
  case $item in
    tests/*) TEST_CASES+=($item)
  esac
  if [[ "$item" == "tests/test_unary_pointwise_ops.py" ]]; then
    UNARY_POINTWISE=1
  fi

  for item_cpu in "${QUICK_CPU_TESTS[@]}"; do
    if [[ "$item" == "$item_cpu" ]]; then
      TEST_CASES_CPU+=($item)
      break
    fi
  done

done

# Skip tests if no tests file is found
if [ ${#TEST_CASES[@]} -eq 0 ]; then
  exit 0
fi

echo "Running unit tests for ${TEST_CASES[@]}"
# TODO(Qiming): Check if utils test should use a different data file
coverage run --data-file=${ID_SHA}-op -m pytest -s -x ${TEST_CASES[@]}

# Run quick-cpu test if necessary
if [[ ${#TEST_CASES_CPU[@]} -ne 0 ]]; then
  echo "Running quick-cpu mode unit tests for ${TEST_CASES_CPU[@]}"
  coverage run --data-file=${ID_SHA}-op -m pytest -s -x ${TEST_CASES_CPU[@]} --ref=cpu --mode=quick
fi

# For unary pointwise operators, adds an additional `abs` mode test
# TODO(Qiming): Check if this special case can be generalized
if [[ "$UNARY_POINTWISE" -eq 1 ]]; then
coverage run --data-file=${ID_SHA}-op -m pytest -s tests/test_unary_pointwise_ops.py -m abs --record=log
fi

# blas_ops=(
#  "tests/test_attention_ops.py"
#  "tests/test_blas_ops.py"
#)

# pointwise_ops=(
#   "tests/test_binary_pointwise_ops.py"
#   "tests/test_pointwise_dynamic.py"
#   "tests/test_pointwise_type_promotion.py"
#   "tests/test_unary_pointwise_ops.py"
# )

# reduction_ops=(
#  "tests/test_general_reduction_ops.py"
#  "tests/test_norm_ops.py"
#  "tests/test_reduction_ops.py"
# )
#
# special_ops=(
#  "tests/test_distribution_ops.py"
#  "tests/test_special_ops.py"
#  "tests/test_tensor_constructor_ops.py"
# )
#
# utils=(
#  "tests/test_libentry.py"
#  "tests/test_shape_utils.py"
#  "tests/test_tensor_wrapper.py"
# )
