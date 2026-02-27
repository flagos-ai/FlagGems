#!/bin/bash

set -e
PR_ID=$1
ID_SHA="${PR_ID}-${GITHUB_SHA}"
echo ID_SHA $ID_SHA

echo $PWD
source tools/run_command.sh
COVERAGE_ARGS="--data-file=${ID_SHA}-utils"
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_libentry.py && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_shape_utils.py && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_wrapper.py

mkdir -p /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
mv ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
