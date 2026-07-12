#!/bin/bash

set -e
PR_ID=$1
ID_SHA="${PR_ID}-${GITHUB_SHA::7}"
COVERAGE_ARGS="--data-file=${ID_SHA}-model"
TEST_CASES=(
  "examples/model_bert_test.py"
)

coverage run ${COVERAGE_ARGS} -m pytest -s -x ${TEST_CASES[@]}
