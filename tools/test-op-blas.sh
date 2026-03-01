#!/bin/bash

set -e
PR_ID=$1
ID_SHA="${PR_ID}-${GITHUB_SHA::7}"
TEST_CASES=(
  "tests/test_attention_ops.py"
  "tests/test_blas_ops.py"
)
COVERAGE_ARGS="--data-file=${ID_SHA}-op"

source tools/run_command.sh

run_command coverage run ${COVERAGE_ARGS} -m pytest -s -x ${TEST_CASES[@]}

mkdir -p /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
mv ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
