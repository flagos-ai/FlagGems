#!/bin/bash

set -e

PR_ID=$1

# Leave this for debugging's purpose
echo "PR_ID=${PR_ID}"

COLLECT_COVERAGE=""

if [[ "$CHANGED_FILES" == "__ALL__" ]]; then
  # Replace "__ALL__" with all tests
  CHANGED_FILES=$(find tests -name "test*.py")
  # add options to generate summary report
  EXTRA_OPTS="--md-report"
  EXTRA_OPTS+=" --md-report-verbose=1"
  EXTRA_OPTS+=" --md-report-output=${PR_ID}-summary.md"
  SUFFIX=""
  COLLECT_COVERAGE="yes"
else
  # for per-PR test, fail early
  EXTRA_OPTS="-x"
  SUFFIX="-${GITHUB_SHA::7}"
fi

# Test cases that needs to run quick cpu tests
NO_QUICK_CPU_TESTS=(
  "tests/ks_tests.py"
  "tests/test_conv3d.py"
  "tests/test_convolution_ops.py"
  "tests/test_distribution_ops.py"
  "tests/test_enable_api.py"
  "tests/test_libentry.py"
  "tests/test_pointwise_type_promotion.py"
  "tests/test_quant.py"
  "tests/test_shape_utils.py"
  "tests/test_tensor_wrapper.py"
  "tests/test_vllm_ops.py"
)

# Extract test cases from CHANGED_FILES
TEST_CASES=()
TEST_CASES_CPU=()
for item in $CHANGED_FILES; do
  case $item in
    # tests/test_DSA/*)
    # skip DSA test for now
    # ;;
    tests/test_quant.py)
      # skip because it always fail
      ;;
    tests/*) TEST_CASES+=($item)
  esac

  # filter out tests that do not need quick CPU mode tests
  found=0
  for item_cpu in "${NO_QUICK_CPU_TESTS[@]}"; do
    if [[ "$item" == "$item_cpu" ]]; then
      found=1
      break
    fi
  done
  if (( $found == 0 )); then
    TEST_CASES_CPU+=($item)
  fi
done

# Skip tests if no tests file is found
if [ ${#TEST_CASES[@]} -eq 0 ]; then
  exit 0
fi

# Clear existing coverage data if any
coverage erase

echo "Running unit tests for ${TEST_CASES[@]}"
# TODO(Qiming): Check if utils test should use a different data file
coverage run -m pytest -s ${EXTRA_OPTS} ${TEST_CASES[@]}

# Run quick-cpu test if necessary
if [[ ${#TEST_CASES_CPU[@]} -ne 0 ]]; then
  echo "Running quick-cpu mode unit tests for ${TEST_CASES_CPU[@]}"
  coverage run -m pytest -s ${EXTRA_OPTS} ${TEST_CASES_CPU[@]} --ref=cpu --quick
fi

# Process coverage data only when full-range testing
# Coverage data HTML dumped to `htmlcov/` by default
if [ -n "$COLLECT_COVERAGE" ]; then
  coverage combine
  coverage html
  rm -fr coverage
  mkdir coverage
  mv htmlcov coverage/
  echo "${PR_ID}${SUFFIX::7}" > coverage/COVERAGE_ID
  mv ${PR_ID}-summary.md coverage/ut-summary.md
fi
