#!/bin/bash

set -e

PR_ID=$1

# Replace "__ALL__" with all tests
if [[ "$CHANGED_FILES" == "__ALL__" ]]; then
  CHANGED_FILES=$(find tests -name "test*.py")
  FAIL_EARLY=""
  echo "TIMESTAMP=${PR_ID}"
  SUFFIX=""
else
  FAIL_EARLY="-x"
  echo "PR_ID=${PR_ID}"
  SUFFIX="-${GITHUB_SHA::7}"
fi

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



TEST_CASES=()
TEST_CASES_CPU=()
for item in $CHANGED_FILES; do
  case $item in
    tests/test_DSA/*)
      # skip DSA test for now
      ;;
    tests/test_quant.py)
      # skip
      ;;
    tests/*) TEST_CASES+=($item)
  esac

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

# Clear existing coverage data if any
coverage erase

echo "Running unit tests for ${TEST_CASES[@]}"
# TODO(Qiming): Check if utils test should use a different data file
coverage run -m pytest -s ${FAIL_EARLY} ${TEST_CASES[@]}

# Run quick-cpu test if necessary
if [[ ${#TEST_CASES_CPU[@]} -ne 0 ]]; then
  echo "Running quick-cpu mode unit tests for ${TEST_CASES_CPU[@]}"
  coverage run -m pytest -s ${FAIL_EARLY} ${TEST_CASES_CPU[@]} --ref=cpu --mode=quick
fi

# Process coverage data only when full-range testing
# Coverage data HTML dumped to `htmlcov/` by default
# if [[ "$CHANGED_FILES" == "__ALL__" ]]; then
  coverage combine
  coverage html
  rm -fr coverage
  mkdir coverage
  mv htmlcov coverage/
  echo "${PR_ID}${SUFFIX::7}" > coverage/COVERAGE_ID
  ls coverage
# fi
