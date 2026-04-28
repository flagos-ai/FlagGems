#!/bin/bash

export MUSA_HOME=/usr/local/musa
export PATH=$MUSA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH

uv pip install -e .
uv pip install ".[mthreads,test]"

uv pip install --index ${FLAGOS_PYPI} \
    "torch==2.7.1+musa.4.0.0" \
    "torch_musa==2.7.1" \
    "triton==3.1.0+musa1.4.6" \
    "numpy==1.26.4" \
    "mkl==2024.0.0"

# For the intel math library
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH

# Replace flagtree with Triton if requested
if [ -n "${USE_TRITON}" ]; then
  uv pip uninstall flagtree
  uv pip install --index $FLAGOS_PYPI \
    "triton==3.1.0+musa1.4.6"
else
  # Flagtree libraries are not exposed by default
  export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/triton/_C:$LD_LIBRARY_PATH
fi
