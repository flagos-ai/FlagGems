#!/bin/bash

uv pip install -e .[metax,test]

export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=$MACA_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH
# For FlagTree
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/triton/backends/metax/lib:$LD_LIBRARY_PATH
