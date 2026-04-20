#!/bin/bash

export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=$MACA_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH

uv pip install -e . .[metax,test]

if [ -n "${USE_FLAGTREE}" ]; then
  uv pip uninstall triton
  uv pip install --index ${FLAGOS_PYPI} \
    "flagtree==3.1.0+metax3.5.3.9"

  SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.12/site-packages
  export LD_LIBRARY_PATH=${SITE_PACKAGES}/triton/backends/metax/lib:$LD_LIBRARY_PATH
fi
