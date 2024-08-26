#!/bin/bash

set -e

PR_ID=$1

# For local test
if [ -z "$PR_ID" ]; then
  PR_ID=168
fi
if [ -z "$GITHUB_SHA" ]; then
  GITHUB_SHA=abcdefg
fi
ID_SHA="${PR_ID}-${GITHUB_SHA}"
echo ID_SHA $ID_SHA

PR_ID_DIR="PR${PR_ID}"

COVERAGE_ARGS="--parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests --data-file=${ID_SHA}-model"
cmds=(
  "CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s examples/model_bert_test.py &"
)

declare -a exit_statuses

for cmd in "${cmds[@]}"; do
    eval "$cmd"
done

for job in $(jobs -p); do
    wait $job
    exit_statuses+=($?)
    echo "Task $pid completed with exit status ${exit_statuses[-1]}"
done

echo "Exit statuses of all tasks: ${exit_statuses[@]}"

overall_status=0
for status in "${exit_statuses[@]}"; do
    if [ $status -ne 0 ]; then
        overall_status=1
        break
    fi
done

mkdir -p /PR_Coverage/PR${PR_ID}/${ID_SHA}
mv ${ID_SHA}* /PR_Coverage/PR${PR_ID}/${ID_SHA}

exit $overall_status
