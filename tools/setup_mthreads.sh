#!/bin/bash

export MUSA_HOME=/usr/local/musa
export PATH=$MUSA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH

echo "Setting up mthreads ..."

uv pip install .[mthreads,test]

uv pip uninstall triton
uv pip install --index $FLAGOS_PYPI \
  "flagtree==0.5.0+mthreads3.1"

uv pip list

# For the intel math library
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
# For the Flagtree dynamic library
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/triton/_C:$LD_LIBRARY_PATH
