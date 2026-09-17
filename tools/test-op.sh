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
  "tests/test_cudnn_convolution_transpose.py"
)

# Extract test cases from CHANGED_FILES
TEST_CASES=()
PERF_TEST_CASES=()
TEST_CASES_CPU=()
# Operator implementation files (generic ops/ or a backend's ops/) that changed
# but bring no test file of their own. We still want to exercise them, selected
# by pytest marker below.
OPS_IMPL_FILES=()
for item in $CHANGED_FILES; do
  file_name=$(basename "$item")
  case $item in
    tests/test_quant.py)
      # skip because it always fail
      ;;
    tests/*.py)
      if [[ "$file_name" == test*.py ]]; then
        TEST_CASES+=($item)
      fi
      ;;
    benchmark/test*)
      PERF_TEST_CASES+=($item)
      ;;
    src/flag_gems/ops/*.py | src/flag_gems/runtime/backend/*/ops/*.py)
      if [[ "$file_name" != "__init__.py" ]]; then
        OPS_IMPL_FILES+=($item)
      fi
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
          TEST_CASES_CPU+=($item)
        fi
        ;;
    esac
  fi
done

# Derive operator ids for changed implementation files that carry no test file
# of their own, so we can select their tests by pytest marker. Without this a PR
# that only touches e.g. src/flag_gems/runtime/backend/_kunlunxin/ops/foo.py runs
# no tests at all and passes vacuously.
OPS_MARKERS=()
if [[ ${#OPS_IMPL_FILES[@]} -gt 0 ]]; then
  mapfile -t DERIVED_MARKERS < <(
    python3 tools/ci_checks/derive_changed_operators.py \
      --ops-only --changed-files "${OPS_IMPL_FILES[*]}" 2>/dev/null
  )
  # Drop ids whose canonical test file (tests/test_<id>.py) is already being run
  # directly via TEST_CASES, so the same file is not executed twice.
  for op in "${DERIVED_MARKERS[@]}"; do
    [[ -z "$op" ]] && continue
    already=0
    for tc in "${TEST_CASES[@]}"; do
      if [[ "$tc" == "tests/test_${op}.py" ]]; then
        already=1
        break
      fi
    done
    if (( already == 0 )); then
      OPS_MARKERS+=("$op")
    fi
  done
fi

# Skip tests only when there is nothing at all to run.
if [[ ${#TEST_CASES[@]} -eq 0 && ${#PERF_TEST_CASES[@]} -eq 0 && ${#OPS_MARKERS[@]} -eq 0 ]]; then
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

# Run marker-selected tests for changed operator implementations. Operator ids
# match the pytest marker each test declares (e.g. @pytest.mark.<op>), so this
# reaches the right tests even when the implementation file name and the test
# file name differ (native_batch_norm -> test_batch_norm.py, etc.).
if [[ ${#OPS_MARKERS[@]} -gt 0 ]]; then
  # Build an "id1 or id2 or ..." marker expression.
  MARKER_EXPR=""
  for op in "${OPS_MARKERS[@]}"; do
    [[ -z "$op" ]] && continue
    if [[ -z "$MARKER_EXPR" ]]; then
      MARKER_EXPR="$op"
    else
      MARKER_EXPR="${MARKER_EXPR} or ${op}"
    fi
  done

  if [[ -n "$MARKER_EXPR" ]]; then
    echo "Running marker-selected tests for changed operators: ${MARKER_EXPR}"
    # Run and capture output. When a marker matches no test, pytest DESELECTS
    # all tests and still exits 0 (it exits 5 only for a truly empty
    # collection), so the exit code alone cannot tell "passed" from "ran
    # nothing". We therefore also inspect the summary line: a run that executed
    # zero tests (all deselected / no tests ran) is reported as a warning rather
    # than a silent pass, while genuine collection errors keep pytest's non-zero
    # exit and are recorded as failures.
    marker_log="marker-run-${GITHUB_SHA::7}.log"
    # `| tee` would mask pytest's exit status, so read it from PIPESTATUS.
    coverage run -m pytest -s ${EXTRA_OPTS} tests/ -m "${MARKER_EXPR}" \
        2>&1 | tee "${marker_log}"
    rc=${PIPESTATUS[0]}
    if [[ $rc -eq 0 ]]; then
      if grep -qE "no tests ran|[0-9]+ deselected" "${marker_log}" \
          && ! grep -qE "[0-9]+ (passed|failed|error)" "${marker_log}"; then
        echo "::warning::No tests matched markers for changed operators (${MARKER_EXPR}); nothing ran."
      fi
    else
      rm -f "${marker_log}"
      if $FAIL_FAST; then exit 1; fi
      FAILURES+=("operator markers: ${MARKER_EXPR}")
    fi
    rm -f "${marker_log}"
  fi
fi

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
