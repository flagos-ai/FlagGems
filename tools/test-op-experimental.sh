#!/usr/bin/env bash


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

# Exit on error or pipe failure
set -eo pipefail

CUDA_VISIBLE_DEVICES=6

source_dir="src/flag_gems/experimental_ops"
unit_test_dir="experimental_tests/unit"
performance_test_dir="experimental_tests/performance"

# Let the tests run all operators if CHANGED_FILES is not specified
changed_files=${CHANGED_FILES}
if [[ "$changed_files" == "" ]]; then
  changed_files=$(ls $source_dir)
  changed_files+=$(ls $unit_test_dir)
  changed_files+=$(ls $performance_test_dir)
fi

# Categorize Tests
unit_tests_to_run=""
performance_tests_to_run=""
unit_missing_tests=""
performance_missing_tests=""

for f in $changed_files; do
    # Changes to operator implementation
    if [[ $f == "$source_dir"/*.py ]]; then
        if [[ $(basename "$f") == __*__* ]]; then continue; fi

        base=$(basename "$f" .py)

        # Unit Test
        unit_test_file="$unit_test_dir/${base}_test.py"
        [[ -f "$unit_test_file" ]] && unit_tests_to_run+=" $unit_test_file" || unit_missing_tests+=" $unit_test_file"

        # Performance Test
        performance_test_file="$performance_test_dir/${base}_test.py"
        [[ -f "$performance_test_file" ]] && performance_tests_to_run+=" $performance_test_file" || performance_missing_tests+=" $performance_test_file"

    # Logic for direct test file changes
    elif [[ $f == "$unit_test_dir"/*_test.py && -f "$f" ]]; then
        unit_tests_to_run+=" $f"
    elif [[ $f == "$performance_test_dir"/*_test.py && -f "$f" ]]; then
        performance_tests_to_run+=" $f"
    fi
done

# Error handling for missing test cases
if [[ -n "$unit_missing_tests" ]]; then
    echo "Missing unit tests: $unit_missing_tests"
    exit 1
fi
if [[ -n "$performance_missing_tests" ]]; then
    echo "Missing performance tests: $performance_missing_tests"
    exit 1
fi

if [[ -n "$unit_tests_to_run" ]]; then
    unique_files=$(echo "$unit_tests_to_run" | tr ' ' '\n' | sort -u | xargs)
    pytest -s $unique_files
fi

if [[ -n "$performance_tests_to_run" ]]; then
    unique_files=$(echo "$performance_tests_to_run" | tr ' ' '\n' | sort -u | xargs)
    pytest -s $unique_files
fi
