#!/bin/bash

# This script is provided by the Huawei Ascend CANN toolkit installation.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# TODO: Check if this is necessary
# export TRITON_ALL_BLOCKS_PARALLEL=1

uv pip install -e . .[ascend,test]

uv pip install --index ${FLAGOS_PYPI} \
    "flagtree==0.5.0+ascend3.2" \
    "torch==2.9.0+cpu" \
    "torch-npu==2.9.0"

if [ -n "${USE_FLAGTREE}" ]; then
  uv pip install --index ${FLAGOS_PYPI} \
    flagtree==0.5.0+ascend3.2
else
  uv pip install --index ${FLAGOS_PYPI} \
    triton_ascend==3.2.0
fi
