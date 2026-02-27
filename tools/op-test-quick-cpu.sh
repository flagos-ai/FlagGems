#!/bin/bash

set -e
PR_ID=$1
ID_SHA="${PR_ID}-${GITHUB_SHA}"
echo ID_SHA $ID_SHA

source tools/run_command.sh
COVERAGE_ARGS="--data-file=${ID_SHA}-op"
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_blas_ops.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_reduction_ops.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_general_reduction_ops.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_norm_ops.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_unary_pointwise_ops.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_binary_pointwise_ops.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_special_ops.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_pointwise_type_promotion.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_constructor_ops.py --ref=cpu --mode=quick && \
run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_attention_ops.py --ref=cpu --mode=quick

mkdir -p /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
mv ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
