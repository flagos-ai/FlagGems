#!/bin/bash

set -e
PR_ID=$1
ID_SHA="${PR_ID}-${GITHUB_SHA}"
echo ID_SHA $ID_SHA

source tools/run_command.sh

COVERAGE_ARGS="--data-file=${ID_SHA}-op"

bash tools/pytest_mark_check.sh && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_blas_ops.py && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_attention_ops.py

mkdir -p /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
mv ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
