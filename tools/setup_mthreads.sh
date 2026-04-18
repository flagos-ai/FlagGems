#!/bin/bash

uv pip install -e .[mthreads,test]

# TODO: Drop the following lines
uv pip install -v --index $FLAGOS_PYPI \
  "torch==2.7.1+musa4.0.0"
uv pip install -v --index $FLAGOS_PYPI \
  "torch_musa==2.7.1"

export MUSA_HOME=/usr/local/musa
export PATH=$MUSA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
# For the intel math library
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
# For the Flagtree dynamic library
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/triton/_C:$LD_LIBRARY_PATH
