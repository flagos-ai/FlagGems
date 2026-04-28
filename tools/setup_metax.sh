#!/bin/bash

export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=$MACA_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH

uv pip install -e  .
uv pip install .[metax,test]

uv pip install --index ${FLAGOS_PYPI} \
    "torch==2.8.0+metax3.5.3.9" \
    "torchaudio==2.4.1+metax3.5.3.9" \
    "torchvision==0.15.1+metax3.5.3.9" \
    "flagtree==3.1.0+metax3.5.3.9"

if [ -n "${USE_TRITON}" ]; then
  uv pip uninstall flagtree
  uv pip install --index ${FLAGOS_PYPI} \
    "triton==3.0.0+metax3.5.3.9"
else
  SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.12/site-packages
  export LD_LIBRARY_PATH=${SITE_PACKAGES}/triton/backends/metax/lib:$LD_LIBRARY_PATH
fi
