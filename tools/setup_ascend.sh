#!/bin/bash

# Initialize Ascend environment variables.
# This script is provided by the Huawei Ascend CANN toolkit installation.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# TODO: Check if this is necessary
# export TRITON_ALL_BLOCKS_PARALLEL=1

# The following command will install torch==2.9.0+cpu as well
# uv pip install --index ${FLAGOS_PYPI} \
#  torch-npu==2.9.0 \
#  flagtree==0.5.0+ascend3.2

uv pip install -e .[ascend,test]
