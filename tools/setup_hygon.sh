#!/bin/bash

# Environment setting for DTK-26.04
source /opt/dtk-26.04/env.sh

uv pip install -e . flag_gems[hygon,test]

if [ -n "${USE_FLAGTREE}" ]; then
  uv pip uninstall triton
  uv pip install --index ${FLAGOS_PYPI} \
    "flagtree==0.5.0+hcu3.0"
fi
