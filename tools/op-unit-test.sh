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

COVERAGE_ARGS="--parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests --data-file=${ID_SHA}-op"
cmds=(
   "bash tools/pytest_mark_check.sh &"
   "CUDA_VISIBLE_DEVICES=0 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_blas_ops.py &"
   "CUDA_VISIBLE_DEVICES=1 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_reduction_ops.py &"
   "CUDA_VISIBLE_DEVICES=2 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_general_reduction_ops.py &"
   "CUDA_VISIBLE_DEVICES=3 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_norm_ops.py &"
   "CUDA_VISIBLE_DEVICES=4 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_unary_pointwise_ops.py && \
    CUDA_VISIBLE_DEVICES=4 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_binary_pointwise_ops.py &"
   "CUDA_VISIBLE_DEVICES=5 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_special_ops.py &"
   "CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_pointwise_type_promotion.py && \
    CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_constructor_ops.py && \
    CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_distribution_ops.py && \
    CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_libentry.py && \
    CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_pointwise_dynamic.py && \
    CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_shape_utils.py && \
    CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_wrapper.py && \
    CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_unary_pointwise_ops.py -m abs --record=log &"
   "CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_blas_ops.py --ref=cpu --mode=quick && \
    CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_reduction_ops.py --ref=cpu --mode=quick && \
    CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_general_reduction_ops.py --ref=cpu --mode=quick && \
    CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_norm_ops.py --ref=cpu --mode=quick && \
    CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_unary_pointwise_ops.py --ref=cpu --mode=quick && \
    CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_binary_pointwise_ops.py --ref=cpu --mode=quick && \
    CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_special_ops.py --ref=cpu --mode=quick && \
    CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_pointwise_type_promotion.py --ref=cpu --mode=quick && \
    CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_constructor_ops.py --ref=cpu --mode=quick &"
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
