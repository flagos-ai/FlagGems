#!/usr/bin/env bash

set -euo pipefail

PR_ID="${1:-}"
if [ -z "$PR_ID" ]; then
  echo "Usage: $0 <pr_id>"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CODE_ROOT="${CODE_ROOT:-$DEFAULT_CODE_ROOT}"
CODE_ROOT="$(cd "$CODE_ROOT" && pwd)"
cd "$CODE_ROOT"

DEFAULT_COMPETITION_ROOT="$CODE_ROOT/tools/competition"
COMPETITION_ROOT="${COMPETITION_ROOT:-$DEFAULT_COMPETITION_ROOT}"
COMPETITION_ROOT="$(cd "$COMPETITION_ROOT" && pwd)"

TASKS_YAML="${TASKS_YAML:-$COMPETITION_ROOT/tasks.yaml}"

PR_TITLE="${PR_TITLE:-${GITHUB_PR_TITLE:-}}"
CHANGED_FILES="${CHANGED_FILES:-}"
EXPLICIT_TASK_IDS="${TASK_IDS:-${EXPLICIT_TASK_IDS:-}}"

WARMUP="${WARMUP:-3}"
ITER="${ITER:-10}"
LEVEL="${LEVEL:-core}"
MODE="${MODE:-kernel}"

PR_AUTHOR="${PR_AUTHOR:-${GITHUB_ACTOR:-}}"
COMMIT_SHA="${COMMIT_SHA:-${GITHUB_SHA:-}}"
GITHUB_ID="${GITHUB_ID:-${GITHUB_ACTOR:-}}"
GITHUB_PR_URL="${GITHUB_PR_URL:-}"
UPLOAD_SCORE="${UPLOAD_SCORE:-1}"

COMPETITION_REPO_ROOT="$(cd "$COMPETITION_ROOT/../.." && pwd)"

# Prefer PR code ($CODE_ROOT) on PYTHONPATH even when the authoritative baseline
# is a full repository checkout.
PYTEST_PYTHONPATH="$CODE_ROOT:$COMPETITION_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# If the authoritative repo is checked out under CODE_ROOT (e.g. CODE_ROOT/authoritative),
# use relative paths for pytest arguments to avoid overly long / sensitive log filenames
# generated from invocation args.
REL_COMPETITION_REPO_ROOT="."
if [[ "$COMPETITION_REPO_ROOT" == "$CODE_ROOT/"* ]]; then
  REL_COMPETITION_REPO_ROOT="${COMPETITION_REPO_ROOT#"$CODE_ROOT/"}"
fi

run_competition_py() {
  local script_path="$1"
  shift
  PYTHONPATH="$COMPETITION_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python "$script_path" "$@"
}

resolve_test_spec() {
  local test_spec="$1"
  local file_path="${test_spec%%::*}"
  local suffix=""

  if [[ "$test_spec" == *"::"* ]]; then
    suffix="::${test_spec#*::}"
  fi

  if [ -f "$COMPETITION_REPO_ROOT/$file_path" ]; then
    if [[ "$REL_COMPETITION_REPO_ROOT" != /* ]]; then
      if [[ "$REL_COMPETITION_REPO_ROOT" == "." ]]; then
        printf '%s%s\n' "$file_path" "$suffix"
      else
        printf '%s/%s%s\n' "$REL_COMPETITION_REPO_ROOT" "$file_path" "$suffix"
      fi
    else
      printf '%s%s\n' "$COMPETITION_REPO_ROOT/$file_path" "$suffix"
    fi
    return
  fi

  printf '%s\n' "$test_spec"
}

TASK_IDS_RESOLVED="$(run_competition_py "$COMPETITION_ROOT/task_resolver.py" \
  --tasks-yaml "$TASKS_YAML" \
  --pr-title "$PR_TITLE" \
  --changed-files "$CHANGED_FILES" \
  --explicit "$EXPLICIT_TASK_IDS" \
  --format plain)"

if [ -z "$TASK_IDS_RESOLVED" ]; then
  echo "No task ids resolved. Provide TASK_IDS/EXPLICIT_TASK_IDS, or PR_TITLE/CHANGED_FILES."
  exit 1
fi

read -r -a TASK_ID_LIST <<< "$TASK_IDS_RESOLVED"

echo "=========================================="
echo "Competition Benchmark for PR #${PR_ID}"
echo "Tasks: ${TASK_ID_LIST[*]}"
echo "=========================================="

for TASK_ID in "${TASK_ID_LIST[@]}"; do
  echo ""
  echo "=========================================="
  echo "Task ID: ${TASK_ID}"
  echo "=========================================="

  TASK_NAME="$(run_competition_py "$COMPETITION_ROOT/task_query.py" --tasks-yaml "$TASKS_YAML" --task-id "$TASK_ID" --field name)"
  CORRECTNESS_TESTS_STR="$(run_competition_py "$COMPETITION_ROOT/task_query.py" --tasks-yaml "$TASKS_YAML" --task-id "$TASK_ID" --field correctness_tests)"
  BENCHMARK_TESTS_STR="$(run_competition_py "$COMPETITION_ROOT/task_query.py" --tasks-yaml "$TASKS_YAML" --task-id "$TASK_ID" --field benchmark_tests)"

  if [ -z "$CORRECTNESS_TESTS_STR" ] || [ -z "$BENCHMARK_TESTS_STR" ]; then
    echo "Missing correctness_tests or benchmark_tests for task_id=$TASK_ID"
    exit 1
  fi

  read -r -a CORRECTNESS_TESTS <<< "$CORRECTNESS_TESTS_STR"
  read -r -a BENCHMARK_TESTS <<< "$BENCHMARK_TESTS_STR"

  for i in "${!CORRECTNESS_TESTS[@]}"; do
    CORRECTNESS_TESTS[$i]="$(resolve_test_spec "${CORRECTNESS_TESTS[$i]}")"
  done

  for i in "${!BENCHMARK_TESTS[@]}"; do
    BENCHMARK_TESTS[$i]="$(resolve_test_spec "${BENCHMARK_TESTS[$i]}")"
  done

  CORRECTNESS_PYTEST_ARGS=()
  for test_case in "${CORRECTNESS_TESTS[@]}"; do
    if [[ "$test_case" == tools/competition/* ]]; then
      CORRECTNESS_PYTEST_ARGS=(-p tests.conftest)
      break
    fi
  done

  BENCHMARK_PYTEST_ARGS=()
  for test_case in "${BENCHMARK_TESTS[@]}"; do
    if [[ "$test_case" == tools/competition/* ]]; then
      BENCHMARK_PYTEST_ARGS=(-p benchmark.conftest)
      break
    fi
  done

  LOG_FILE="benchmark_result_pr${PR_ID}_${TASK_ID}.log"
  SCORE_FILE="score_pr${PR_ID}_${TASK_ID}.json"
  CORRECTNESS_XML="correctness_pr${PR_ID}_${TASK_ID}.xml"

  echo ""
  echo "[Task ${TASK_ID}] Running correctness tests"
  echo "Tests: ${CORRECTNESS_TESTS[*]}"

  CORRECTNESS_EXIT=0
  PYTHONPATH="$PYTEST_PYTHONPATH" pytest -v "${CORRECTNESS_PYTEST_ARGS[@]}" --junitxml="$CORRECTNESS_XML" "${CORRECTNESS_TESTS[@]}" || CORRECTNESS_EXIT=$?

  # Extract passed/total from junitxml
  CORRECTNESS_STATS="$(python -c "
import xml.etree.ElementTree as ET, sys
try:
    root = ET.parse('$CORRECTNESS_XML').getroot()
    tests = int(root.attrib.get('tests', 0))
    failures = int(root.attrib.get('failures', 0))
    errors = int(root.attrib.get('errors', 0))
    skipped = int(root.attrib.get('skipped', 0))
    passed = tests - failures - errors - skipped
    print(f'{passed} {tests}')
except Exception:
    print('0 0')
")"
  read -r CORRECTNESS_PASSED CORRECTNESS_TOTAL <<< "$CORRECTNESS_STATS"
  echo "[Task ${TASK_ID}] Correctness: ${CORRECTNESS_PASSED}/${CORRECTNESS_TOTAL} passed"

  if [ "$CORRECTNESS_EXIT" -ne 0 ]; then
    echo "Correctness tests failed for task_id=$TASK_ID"

    run_competition_py "$COMPETITION_ROOT/calculate_competition_score.py" \
      --output "$SCORE_FILE" \
      --pr-id "$PR_ID" \
      --pr-title "$PR_TITLE" \
      --pr-author "$PR_AUTHOR" \
      --commit-sha "$COMMIT_SHA" \
      --correctness-failed \
      --correctness-passed "$CORRECTNESS_PASSED" \
      --correctness-total "$CORRECTNESS_TOTAL"

    SCORE="$(python -c "import json; print(json.load(open(r'$SCORE_FILE', 'r', encoding='utf-8'))['total_score'])")"
  else
    echo "Correctness passed for task_id=$TASK_ID"

    rm -f result_*.log 2>/dev/null || true

    echo ""
    echo "[Task ${TASK_ID}] Running benchmark tests"
    echo "Parameters: warmup=$WARMUP iter=$ITER level=$LEVEL mode=$MODE"
    echo "Tests: ${BENCHMARK_TESTS[*]}"

    PYTHONPATH="$PYTEST_PYTHONPATH" pytest -v "${BENCHMARK_PYTEST_ARGS[@]}" \
      --warmup="$WARMUP" \
      --iter="$ITER" \
      --level="$LEVEL" \
      --mode="$MODE" \
      --record=log \
      "${BENCHMARK_TESTS[@]}"

    LATEST_LOG="$(ls -t result_*.log 2>/dev/null | head -1 || true)"
    if [ -z "$LATEST_LOG" ]; then
      echo "ERROR: No benchmark log file generated"
      exit 1
    fi

    mv "$LATEST_LOG" "$LOG_FILE"

    run_competition_py "$COMPETITION_ROOT/calculate_competition_score.py" \
      --log "$LOG_FILE" \
      --output "$SCORE_FILE" \
      --pr-id "$PR_ID" \
      --pr-title "$PR_TITLE" \
      --pr-author "$PR_AUTHOR" \
      --commit-sha "$COMMIT_SHA" \
      --correctness-passed "$CORRECTNESS_PASSED" \
      --correctness-total "$CORRECTNESS_TOTAL"

    SCORE="$(python -c "import json; print(json.load(open(r'$SCORE_FILE', 'r', encoding='utf-8'))['total_score'])")"
  fi

  echo "[Task ${TASK_ID}] Score file: $SCORE_FILE"
  echo "[Task ${TASK_ID}] Total score: $SCORE"

  # Extract score_details JSON for upload
  SCORE_DETAILS="$(python -c "import json; print(json.dumps(json.load(open(r'$SCORE_FILE', 'r', encoding='utf-8')).get('score_details', {})))")"

  if [ "$UPLOAD_SCORE" = "1" ]; then
    run_competition_py "$COMPETITION_ROOT/upload_score.py" \
      --name "$TASK_NAME" \
      --github-id "$GITHUB_ID" \
      --github-pr "$GITHUB_PR_URL" \
      --score "$SCORE" \
      --score-details "$SCORE_DETAILS" \
      --note "pr_id=$PR_ID task_id=$TASK_ID"
  fi
done

echo ""
echo "All tasks completed: ${TASK_ID_LIST[*]}"
