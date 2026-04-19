#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH

uv pip install -e . .[iluvatar,test]

if [ -z "${USE_FLAGTREE}" ]; then
  uv pip install --index $FLAGOS_PYPI \
    "triton==3.1.0+corex.4.4.0"
fi
