#!/usr/bin/env bash

set -eo pipefail  # Exit on error or pipe failure

source tools/run_command.sh

# 1. Environment Check
if [[ -z "$BASE_SHA" || -z "$HEAD_SHA" ]]; then
    echo "Error: BASE_SHA or HEAD_SHA is not set."
    exit 1
fi

echo "Diffing $BASE_SHA...$HEAD_SHA"

# 2. Identify Changed Files
changed_files=$(git diff --name-only --diff-filter=AM "$BASE_SHA...$HEAD_SHA" || true)

if [[ -z "$changed_files" ]]; then
    echo "No changed files detected, skipping."
    exit 0
fi

# 3. Categorize Tests
unit_tests_to_run=""
performance_tests_to_run=""
unit_missing_tests=""
performance_missing_tests=""

source_dir="${source_dir:-src/flag_gems/experimental_ops}"
unit_test_dir="${unit_test_dir:-experimental_tests/unit}"
performance_test_dir="${performance_test_dir:-experimental_tests/performance}"

for f in $changed_files; do
    # Logic for files in experimental_ops
    if [[ $f == "$source_dir"/*.py ]]; then
        if [[ $(basename "$f") == __*__* ]]; then continue; fi

        base=$(basename "$f" .py)

        # Check Unit Test
        unit_test_file="$unit_test_dir/${base}_test.py"
        [[ -f "$unit_test_file" ]] && unit_tests_to_run+=" $unit_test_file" || unit_missing_tests+=" $unit_test_file"

        # Check Performance Test
        performance_test_file="$performance_test_dir/${base}_test.py"
        [[ -f "$performance_test_file" ]] && performance_tests_to_run+=" $performance_test_file" || performance_missing_tests+=" $performance_test_file"

    # Logic for direct test file changes
    elif [[ $f == "$unit_test_dir"/*_test.py && -f "$f" ]]; then
        unit_tests_to_run+=" $f"
    elif [[ $f == "$performance_test_dir"/*_test.py && -f "$f" ]]; then
        performance_tests_to_run+=" $f"
    fi
done

# 4. Error Handling for Missing Tests
if [[ -n "$unit_missing_tests" || -n "$performance_missing_tests" ]]; then
    echo "::error:: Modified operators are missing required test files."
    [[ -n "$unit_missing_tests" ]] && echo "Missing Unit: $unit_missing_tests"
    [[ -n "$performance_missing_tests" ]] && echo "Missing Performance: $performance_missing_tests"
    exit 1
fi

# 5. Execution Helper
run_pytest_group() {
    local label=$1
    local files=$2
    if [[ -n "$files" ]]; then
        unique_files=$(echo "$files" | tr ' ' '\n' | sort -u | xargs)
        echo "Running $label tests: $unique_files"
        run_command pytest -s $unique_files
    else
        echo "No $label tests to run."
    fi
}

run_pytest_group "Unit" "$unit_tests_to_run"
run_pytest_group "Performance" "$performance_tests_to_run"
