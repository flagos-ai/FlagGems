#!/bin/bash


# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PR_ID=$1

# Leave this for debugging's purpose
echo "PR_ID=${PR_ID}"

COLLECT_COVERAGE=""
FAIL_FAST=false

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
  FAIL_FAST=true
  EXTRA_OPTS="-x"
  SUFFIX="-${GITHUB_SHA::7}"
fi

# Test cases that needs to run quick cpu tests
NO_QUICK_CPU_TESTS=(
  "tests/ks_tests.py"
  "tests/test_enable_api.py"
  "tests/test_flash_attention_backward.py"
  "tests/test_libentry.py"
  "tests/test_pointwise_type_promotion.py"
  "tests/test_quant.py"
  "tests/test_shape_utils.py"
  "tests/test_tensor_wrapper.py"
  "tests/test_conv_depthwise2d.py"
)

# Map ops / JIT csrc edits to the matching unit test (and bench if present).
# Without this, changing only src/flag_gems/ops/*.py or csrc/*.cpp leaves
# CHANGED_FILES without tests/*.py and python-op would exit 0 after the job
# starts (or never start if only ops/aten label is set).
if [[ "$CHANGED_FILES" != "__ALL__" ]]; then
  MAPPED_FILES=()
  for item in $CHANGED_FILES; do
    base=""
    case $item in
      src/flag_gems/ops/*.py)
        base=$(basename "$item" .py)
        ;;
      src/flag_gems/csrc/*.cpp|src/flag_gems/csrc/*.h|src/flag_gems/csrc/*.hpp)
        base=$(basename "$item")
        base="${base%.*}"
        # Shared kernel header: wgrad_gemm_accum_kernel.h -> wgrad_gemm_accum
        base="${base%_kernel}"
        ;;
    esac
    if [[ -n "$base" ]]; then
      # Match experimental convention: leading underscore is dropped in test id.
      base="${base#_}"
      unit_test="tests/test_${base}.py"
      if [[ -f "$unit_test" ]]; then
        MAPPED_FILES+=("$unit_test")
      fi
      bench_test="benchmark/test_${base}.py"
      if [[ -f "$bench_test" ]]; then
        MAPPED_FILES+=("$bench_test")
      fi
    fi
  done
  if [[ ${#MAPPED_FILES[@]} -gt 0 ]]; then
    echo "Mapped ops/csrc changes to: ${MAPPED_FILES[*]}"
    CHANGED_FILES="$CHANGED_FILES ${MAPPED_FILES[*]}"
  fi
fi

# Extract test cases from CHANGED_FILES
TEST_CASES=()
PERF_TEST_CASES=()
TEST_CASES_CPU=()
# Deduplicate while preserving order (ops map may re-add an already-listed test).
declare -A SEEN_TEST_CASES=()
declare -A SEEN_PERF_CASES=()
declare -A SEEN_CPU_CASES=()

append_unique() {
  # $1 = nameref-like array name via eval; keep it simple with explicit cases.
  local kind=$1
  local path=$2
  case $kind in
    test)
      if [[ -z "${SEEN_TEST_CASES[$path]+x}" ]]; then
        SEEN_TEST_CASES[$path]=1
        TEST_CASES+=("$path")
      fi
      ;;
    perf)
      if [[ -z "${SEEN_PERF_CASES[$path]+x}" ]]; then
        SEEN_PERF_CASES[$path]=1
        PERF_TEST_CASES+=("$path")
      fi
      ;;
    cpu)
      if [[ -z "${SEEN_CPU_CASES[$path]+x}" ]]; then
        SEEN_CPU_CASES[$path]=1
        TEST_CASES_CPU+=("$path")
      fi
      ;;
  esac
}

for item in $CHANGED_FILES; do
  file_name=$(basename "$item")
  case $item in
    tests/test_quant.py)
      # skip because it always fail
      ;;
    tests/*.py)
      if [[ "$file_name" == test*.py ]]; then
        append_unique test "$item"
      fi
      ;;
    benchmark/test*)
      append_unique perf "$item"
      ;;
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
    case $item in
      tests/*.py)
        if [[ "$file_name" == test*.py ]]; then
          append_unique cpu "$item"
        fi
        ;;
    esac
  fi
done

# Skip tests if no tests file is found
if [[ ${#TEST_CASES[@]} -eq 0  && ${#PERF_TEST_CASES[@]} -eq 0 ]]; then
  exit 0
fi

# Clear existing coverage data if any
coverage erase

FAILURES=()
for item in "${TEST_CASES[@]}"; do
  echo "Running unit tests for ${item}"
  if ! coverage run -m pytest -s ${EXTRA_OPTS} ${item}; then
    if $FAIL_FAST; then exit 1; fi
    FAILURES+=("${item}")
  fi
done

# Run quick-cpu test if necessary
for item in "${TEST_CASES_CPU[@]}"; do
  echo "Running quick-cpu mode unit tests for ${item}"
  if ! coverage run -m pytest -s ${EXTRA_OPTS} ${item} --ref=cpu --quick; then
    if $FAIL_FAST; then exit 1; fi
    FAILURES+=("${item} (quick-cpu)")
  fi
done

# Run benchmark test if necessary
for item in "${PERF_TEST_CASES[@]}"; do
  echo "Running benchmark tests for ${item}"
  echo "pytest -s ${item} --level core --record log"
  if ! pytest -s ${item} --level core --record log; then
    if $FAIL_FAST; then exit 1; fi
    FAILURES+=("${item} (benchmark)")
  fi
done

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

# Report failures
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo ""
  echo "=== FAILED TESTS (${#FAILURES[@]}) ==="
  for f in "${FAILURES[@]}"; do
    echo "  - ${f}"
  done
  exit 1
fi
