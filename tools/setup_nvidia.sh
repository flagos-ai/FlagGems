#!/bin/bash

uv pip install -e . .[nvidia,test]

if [ -n "${USE_FLAGTREE}" ]; then
  uv pip uninstall triton
  uv pip install --index ${FLAGOS_PYPI} \
    flagtree==0.5.0+3.5
fi
