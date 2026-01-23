#!/bin/bash

set -e
PR_ID=$1; shift
TESTS=("$@")
ID_SHA="${PR_ID}-${GITHUB_SHA}"

source tools/run_command.sh
COVERAGE_ARGS="--parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests --data-file=coverage-${ID_SHA}"
for CASE in "${TESTS[@]}" ; do
  run_command coverage run ${COVERAGE_ARGS} -m pytest -s ${CASE} --ref=cpu --mode=quick
done

# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_reduction_ops.py --ref=cpu --mode=quick && \
# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_general_reduction_ops.py --ref=cpu --mode=quick && \
# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_norm_ops.py --ref=cpu --mode=quick && \
# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_unary_pointwise_ops.py --ref=cpu --mode=quick && \
# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_binary_pointwise_ops.py --ref=cpu --mode=quick && \
# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_special_ops.py --ref=cpu --mode=quick && \
# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_pointwise_type_promotion.py --ref=cpu --mode=quick && \
# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_constructor_ops.py --ref=cpu --mode=quick && \
# run_command coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_attention_ops.py --ref=cpu --mode=quick
