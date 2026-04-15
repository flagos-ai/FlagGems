#!/bin/bash

# Override triton
uv pip uninstall triton
uv pip install --index ${FLAGOS_PYPI} \
    flagtree==0.5.0+3.5

uv pip install -e .[nvidia,test]
