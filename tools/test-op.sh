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
RUN_KERNEL_CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-check) RUN_KERNEL_CHECK=1 ;;
    *) PR_ID="$1" ;;
  esac
  shift
done

# Ensure console scripts installed by 'pip install .[test]' (coverage, pytest,
# pytest-md-report, …) are on PATH. The native backend gets this automatically
# from 'source .venv/bin/activate' in the workflow; containerized backends
# (kunlunxin, metax, hygon, iluvatar, mthreads, thead, enflame) install into
# the container's system Python without activating anything, so the scripts
# directory (e.g. /flagos/bin) may be absent from PATH under bash --login.
# sysconfig.get_path("scripts") returns the directory for whatever python3 is
# currently in PATH, covering both venv and system installs correctly.
_py_scripts=$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))')
case ":${PATH}:" in
  *":${_py_scripts}:"*) ;;  # already present, nothing to do
  *) export PATH="${_py_scripts}:${PATH}" ;;
esac
unset _py_scripts

# Leave this for debugging's purpose
echo "PR_ID=${PR_ID}"

COLLECT_COVERAGE=""
FAIL_FAST=false
# Remember whether this is a full-range run: CHANGED_FILES is overwritten with
# the discovered test list below, so it cannot be compared with "__ALL__" later.
FULL_RUN=false

if [[ "$CHANGED_FILES" == "__ALL__" ]]; then
  FULL_RUN=true
  # Replace "__ALL__" with all tests
  CHANGED_FILES=$(find tests -name "test*.py")
  # add options to generate summary report
  EXTRA_OPTS="--md-report"
  EXTRA_OPTS+=" --md-report-verbose=1"
  # Reduce test quantity when QUICK_TESTS is set (e.g. daily workflow)
  if [[ -n "${QUICK_TESTS:-}" ]]; then
    EXTRA_OPTS+=" --first-parameter-only"
  fi
  # --md-report-output is set per-batch (see run_batch); the final file is
  # assembled by concatenating all per-batch reports at the end.
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
  "tests/test_rms_norm_w8a16_fp8.py"
  "tests/test_shape_utils.py"
  "tests/test_tensor_wrapper.py"
  "tests/test_conv_depthwise2d.py"
  "tests/test_conv_depthwise3d.py"
  "tests/test_cudnn_convolution_transpose.py"
)

# Benchmark test cases that should not run on generic NVIDIA/CPU environments
# (vendor-specific or hardware-specific tests)
NO_KERNEL_CHECK_TESTS=(
  "benchmark/test_rms_norm_w8a16_fp8.py"
)

# Extract test cases from CHANGED_FILES
TEST_CASES=()
PERF_TEST_CASES=()
TEST_CASES_CPU=()
# Operator implementation files (generic ops/ or a backend's ops/) that changed
# but bring no test file of their own. We still want to exercise them, selected
# by pytest marker below.
OPS_IMPL_FILES=()
declare -A DAILY_PYTHON_OP_BLACKLIST=()
if $FULL_RUN && [[ "${DAILY_PYTHON_OP:-}" == "1" ]]; then
  while read -r test_file _reason; do
    [[ -z "$test_file" || "$test_file" == \#* ]] && continue
    DAILY_PYTHON_OP_BLACKLIST["$test_file"]=1
  done < "$(dirname "${BASH_SOURCE[0]}")/daily-python-op-blacklist.txt"
fi
declare -A ASCEND_DAILY_OP_BLACKLIST=()
if $FULL_RUN && [[ "${BACKEND:-}" == *ascend* ]]; then
  while read -r test_file _reason; do
    [[ -z "$test_file" || "$test_file" == \#* ]] && continue
    ASCEND_DAILY_OP_BLACKLIST["$test_file"]=1
  done < "$(dirname "${BASH_SOURCE[0]}")/ascend-daily-op-blacklist.txt"
fi
for item in $CHANGED_FILES; do
  if [[ -n "${DAILY_PYTHON_OP_BLACKLIST[$item]:-}" ]]; then
    echo "Skipping daily python-op blacklist: $item"
    continue
  fi
  if [[ -n "${ASCEND_DAILY_OP_BLACKLIST[$item]:-}" ]]; then
    # skip on Ascend daily run: CANN 9.1.1 / BiShengHIR compiler limitation
    # TODO: unblock each test once the underlying issue is resolved
    echo "Skipping Ascend daily-op blacklist: $item"
    continue
  fi
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

# Derive pytest markers for changed implementation files that carry no test
# file of their own, so we can select their tests by marker. Without this a PR
# that only touches e.g. src/flag_gems/runtime/backend/_kunlunxin/ops/foo.py
# runs no tests at all and passes vacuously. The derive script applies the
# #6359 marker convention (e.g. _pad_enum -> pad_enum), so the markers here
# match those declared in the test files.
OPS_MARKERS=()
if [[ ${#OPS_IMPL_FILES[@]} -gt 0 ]]; then
  mapfile -t DERIVED_MARKERS < <(
    python3 tools/ci_checks/derive_changed_operators.py \
      --ops-only --changed-files "${OPS_IMPL_FILES[*]}" 2>/dev/null
  )
  # Drop markers whose test file (tests/test_<marker>.py) is already being run
  # directly via TEST_CASES, so the same file is not executed twice. The marker
  # already has the leading underscore stripped, so this matches the test file
  # naming (marker pad_enum -> tests/test_pad_enum.py).
  for marker in "${DERIVED_MARKERS[@]}"; do
    [[ -z "$marker" ]] && continue
    already=0
    for tc in "${TEST_CASES[@]}"; do
      if [[ "$tc" == "tests/test_${marker}.py" ]]; then
        already=1
        break
      fi
    done
    if (( already == 0 )); then
      OPS_MARKERS+=("$marker")
    fi
  done
fi

# Sharding: when TOTAL_SHARDS and SHARD_INDEX are set (e.g. by the daily
# workflow matrix), slice TEST_CASES and TEST_CASES_CPU so each parallel job
# handles a disjoint subset.  Round-robin assignment keeps shard sizes equal.
# OPS_MARKERS are not sharded: they only arise in per-PR runs (CHANGED_FILES !=
# __ALL__), where TOTAL_SHARDS is never set.
if [[ -n "${TOTAL_SHARDS:-}" && -n "${SHARD_INDEX:-}" ]]; then
  sharded=()
  total=${#TEST_CASES[@]}
  for (( i=SHARD_INDEX; i<total; i+=TOTAL_SHARDS )); do
    sharded+=("${TEST_CASES[$i]}")
  done
  TEST_CASES=("${sharded[@]}")

  sharded_cpu=()
  total_cpu=${#TEST_CASES_CPU[@]}
  for (( i=SHARD_INDEX; i<total_cpu; i+=TOTAL_SHARDS )); do
    sharded_cpu+=("${TEST_CASES_CPU[$i]}")
  done
  TEST_CASES_CPU=("${sharded_cpu[@]}")
fi

# Skip tests only when there is nothing at all to run.
if [[ ${#TEST_CASES[@]} -eq 0 && ${#PERF_TEST_CASES[@]} -eq 0 && ${#OPS_MARKERS[@]} -eq 0 ]]; then
  exit 0
fi

# Clear existing coverage data if any
coverage erase

# Run a batch of test files inside a single pytest process. Batching amortizes
# the per-process startup cost of importing flag_gems (hundreds of submodules,
# ~125s on the CI runner) across many files, cutting process startups from
# ~N to ~N/BATCH_SIZE. A shell-level timeout guards against tests that hang
# (e.g. an op that never returns), since pytest-timeout is not a dependency; a
# timed-out batch loses its own coverage data but leaves other batches intact.
#
#   run_batch <label> <report_file|-> <pytest_flags> <file1> <file2> ...
#
# Pass "-" as report_file to omit --md-report-output (e.g. per-PR runs where
# --md-report is not in flags at all).
#
# Per-batch report files are collected in BATCH_REPORTS and concatenated into
# ${PR_ID}-summary.md at the end of the daily run.
BATCH_REPORTS=()
run_batch() {
  local label="$1"; shift
  local report_file="$1"; shift
  local flags="$1"; shift
  local timeout_s="${TEST_BATCH_TIMEOUT:-3600}"
  local extra_flags=""
  if [[ "$report_file" != "-" ]]; then
    extra_flags="--md-report-output=${report_file}"
    BATCH_REPORTS+=("${report_file}")
  fi
  echo "Running ${label} for batch of $# files:"
  printf '  %s\n' "$@"
  timeout "$timeout_s" coverage run -m pytest -s $flags $extra_flags "$@"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    if [[ $rc -eq 124 ]]; then
      echo "::error::${label} batch timed out after ${timeout_s}s"
      FAILURES+=("${label} batch (timeout, $# files): $*")
    else
      FAILURES+=("${label} batch ($# files): $*")
    fi
  fi
}

FAILURES=()
_BATCH_IDX=0
if $FULL_RUN && [[ ${#TEST_CASES[@]} -gt 0 ]]; then
  # Full-range daily run: batch files so each pytest process covers many files.
  BATCH_SIZE="${TEST_BATCH_SIZE:-100}"
  batch=()
  for item in "${TEST_CASES[@]}"; do
    batch+=("$item")
    if [[ ${#batch[@]} -ge $BATCH_SIZE ]]; then
      run_batch "unit tests" "${PR_ID}-summary-batch${_BATCH_IDX}.md" "${EXTRA_OPTS}" "${batch[@]}"
      (( _BATCH_IDX++ )) || true
      batch=()
    fi
  done
  if [[ ${#batch[@]} -gt 0 ]]; then
    run_batch "unit tests" "${PR_ID}-summary-batch${_BATCH_IDX}.md" "${EXTRA_OPTS}" "${batch[@]}"
    (( _BATCH_IDX++ )) || true
  fi
else
  # Per-PR run: a handful of changed files, run individually for precise
  # failure attribution and -x fail-fast. Guard each file with the same shell
  # timeout as the batched daily runs, and print node ids, so a hung test fails
  # with rc=124 and leaves the last started test in the log instead of blocking
  # the job until the workflow timeout.
  for item in "${TEST_CASES[@]}"; do
    echo "Running unit tests for ${item}"
    PYTHONUNBUFFERED=1 timeout "${TEST_BATCH_TIMEOUT:-3600}" \
      coverage run -m pytest -s -v ${EXTRA_OPTS} ${item}
    rc=$?
    if [[ $rc -ne 0 ]]; then
      if [[ $rc -eq 124 ]]; then
        echo "::error::${item} timed out after ${TEST_BATCH_TIMEOUT:-3600}s"
      fi
      if $FAIL_FAST; then exit 1; fi
      FAILURES+=("${item}")
    fi
  done
fi

# Run marker-selected tests for changed operator implementations that have no
# test file of their own. Markers follow the #6359 convention, so this reaches
# the right tests even when the implementation file name differs from the test
# file name (e.g. _pad_enum -> marker pad_enum -> tests/test_pad_enum.py).
if [[ ${#OPS_MARKERS[@]} -gt 0 ]]; then
  # Build an "m1 or m2 or ..." marker expression.
  MARKER_EXPR=""
  for marker in "${OPS_MARKERS[@]}"; do
    [[ -z "$marker" ]] && continue
    if [[ -z "$MARKER_EXPR" ]]; then
      MARKER_EXPR="$marker"
    else
      MARKER_EXPR="${MARKER_EXPR} or ${marker}"
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
      rm -f "${marker_log}"
    else
      rm -f "${marker_log}"
      if $FAIL_FAST; then exit 1; fi
      FAILURES+=("operator markers: ${MARKER_EXPR}")
    fi
  fi
fi

# Run quick-cpu test if necessary.
# Skipped for full-range daily runs (CHANGED_FILES == __ALL__) because the GPU
# accuracy run already covers all files; running them again on CPU reference
# doubles the runtime for negligible extra signal in that context.
if ! $FULL_RUN; then
  for item in "${TEST_CASES_CPU[@]}"; do
    echo "Running quick-cpu mode unit tests for ${item}"
    if ! coverage run -m pytest -s ${EXTRA_OPTS} ${item} --ref=cpu --quick; then
      if $FAIL_FAST; then exit 1; fi
      FAILURES+=("${item} (quick-cpu)")
    fi
  done
fi

# Run benchmark test if necessary
for item in "${PERF_TEST_CASES[@]}"; do
  echo "Running benchmark tests for ${item}"
  echo "pytest -s ${item} --level core --record log"
  if ! pytest -s ${item} --level core --record log; then
    if $FAIL_FAST; then exit 1; fi
    FAILURES+=("${item} (benchmark)")
  fi
done

# Derive benchmark files for changed operators (from unit-test files and
# marker-derived ops) that were NOT already covered by PERF_TEST_CASES above.
# This ensures that a PR adding/modifying an operator implementation or its
# unit tests still gets a kernel-check even when the PR does not touch the
# benchmark file itself.
# kernel-check is opt-in: set RUN_KERNEL_CHECK=1 to enable (e.g. python-op /
# NVIDIA CI). Backend jobs leave it unset and skip this block entirely.
if [[ "$RUN_KERNEL_CHECK" == "1" ]]; then
KERNEL_CHECK_EXTRAS=()
_already_checked=" ${PERF_TEST_CASES[*]} "
for item in "${TEST_CASES[@]}"; do
  bench="benchmark/$(basename "$item")"
  if [[ -f "$bench" ]] && [[ ! "$_already_checked" =~ " ${bench} " ]]; then
    KERNEL_CHECK_EXTRAS+=("$bench")
    _already_checked+="${bench} "
  fi
done
for marker in "${OPS_MARKERS[@]}"; do
  [[ -z "$marker" ]] && continue
  bench="benchmark/test_${marker}.py"
  if [[ -f "$bench" ]] && [[ ! "$_already_checked" =~ " ${bench} " ]]; then
    KERNEL_CHECK_EXTRAS+=("$bench")
    _already_checked+="${bench} "
  fi
done

for item in "${KERNEL_CHECK_EXTRAS[@]}"; do
  # Skip vendor-specific/hardware-specific benchmark tests
  skip_kernel_check=0
  for skip_test in "${NO_KERNEL_CHECK_TESTS[@]}"; do
    if [[ "$item" == "$skip_test" ]]; then
      skip_kernel_check=1
      break
    fi
  done

  if [[ $skip_kernel_check -eq 1 ]]; then
    echo "Skipping kernel-check for ${item} (vendor-specific test)"
    continue
  fi

  echo "Running kernel-check for ${item} (derived from changed operator)"
  if ! pytest -s ${item} --kernel-check; then
    if $FAIL_FAST; then exit 1; fi
    FAILURES+=("${item} (kernel-check)")
  fi
done
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
  # Concatenate per-batch md-report files into the single summary the
  # downstream step expects. Each file starts with a markdown header so the
  # combined file is still valid; we just append them in order.
  if [[ ${#BATCH_REPORTS[@]} -gt 0 ]]; then
    cat "${BATCH_REPORTS[@]}" > "${PR_ID}-summary.md"
    rm -f "${BATCH_REPORTS[@]}"
  fi
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
