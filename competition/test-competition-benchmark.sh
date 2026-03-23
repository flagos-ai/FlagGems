#!/usr/bin/env bash

set -euo pipefail

PR_ID="${1:-}"
if [ -z "$PR_ID" ]; then
  echo "Usage: $0 <pr_id>"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-$DEFAULT_CODE_ROOT}"
CODE_ROOT="$(cd "$CODE_ROOT" && pwd)"
cd "$CODE_ROOT"

DEFAULT_COMPETITION_ROOT="$CODE_ROOT/competition"
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

COMPETITION_REPO_ROOT="$(cd "$COMPETITION_ROOT/.." && pwd)"
PYTEST_PYTHONPATH="$COMPETITION_REPO_ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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
    printf '%s%s\n' "$COMPETITION_REPO_ROOT/$file_path" "$suffix"
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

  LOG_FILE="benchmark_result_pr${PR_ID}_${TASK_ID}.log"
  SCORE_FILE="score_pr${PR_ID}_${TASK_ID}.json"

  echo ""
  echo "[Task ${TASK_ID}] Running correctness tests"
  echo "Tests: ${CORRECTNESS_TESTS[*]}"

  if ! PYTHONPATH="$PYTEST_PYTHONPATH" pytest -v -x "${CORRECTNESS_TESTS[@]}"; then
    echo "Correctness tests failed for task_id=$TASK_ID"

    run_competition_py "$COMPETITION_ROOT/calculate_competition_score.py" \
      --output "$SCORE_FILE" \
      --pr-id "$PR_ID" \
      --pr-title "$PR_TITLE" \
      --pr-author "$PR_AUTHOR" \
      --commit-sha "$COMMIT_SHA" \
      --correctness-failed

    SCORE="0"
  else
    echo "Correctness passed for task_id=$TASK_ID"

    rm -f result_*.log 2>/dev/null || true

    echo ""
    echo "[Task ${TASK_ID}] Running benchmark tests"
    echo "Parameters: warmup=$WARMUP iter=$ITER level=$LEVEL mode=$MODE"
    echo "Tests: ${BENCHMARK_TESTS[*]}"

    PYTHONPATH="$PYTEST_PYTHONPATH" pytest -v \
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
      --commit-sha "$COMMIT_SHA"

    SCORE="$(python -c "import json; print(json.load(open(r'$SCORE_FILE', 'r', encoding='utf-8'))['performance']['total_score'])")"
  fi

  echo "[Task ${TASK_ID}] Score file: $SCORE_FILE"
  echo "[Task ${TASK_ID}] Total score: $SCORE"

  if [ "$UPLOAD_SCORE" = "1" ]; then
    run_competition_py "$COMPETITION_ROOT/upload_score.py" \
      --name "$TASK_NAME" \
      --github-id "$GITHUB_ID" \
      --github-pr "$GITHUB_PR_URL" \
      --score "$SCORE" \
      --note "pr_id=$PR_ID task_id=$TASK_ID"
  fi
done

echo ""
echo "All tasks completed: ${TASK_ID_LIST[*]}"
