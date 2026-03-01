#!/bin/bash

set -e
PR_ID=$1
ID_SHA="${PR_ID}-${GITHUB_SHA::7}"
COVERAGE_ARGS="--data-file=${ID_SHA}-op"
TEST_CASES=(
  "tests/test_attention_ops.py"
  "tests/test_binary_pointwise_ops.py"
  "tests/test_blas_ops.py"
  "tests/test_general_reduction_ops.py"
  "tests/test_norm_ops.py"
  "tests/test_pointwise_type_promotion.py"
  "tests/test_reduction_ops.py"
  "tests/test_special_ops.py"
  "tests/test_tensor_constructor_ops.py"
  "tests/test_unary_pointwise_ops.py"
)

source tools/run_command.sh

run_command coverage run ${COVERAGE_ARGS} -m pytest -s -x ${TEST_CASES[@]} --ref=cpu --mode=quick

mkdir -p /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
mv ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
