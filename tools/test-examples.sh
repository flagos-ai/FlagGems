#!/bin/bash

set -e
PR_ID=$1
ID_SHA="${PR_ID}-${GITHUB_SHA::7}"
COVERAGE_ARGS="--data-file=${ID_SHA}-model"
TEST_CASES=(
  "examples/model_bert_test.py"
)

source tools/run_command.sh

run_command coverage run ${COVERAGE_ARGS} -m pytest -s -x ${TEST_CASES[@]}

mkdir -p /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
mv ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
