#!/bin/bash

# Environment setting for DTK-26.04
source /opt/dtk-26.04/env.sh

uv pip install -e . .[hygon,test]

uv pip install --index ${FLAGOS_PYPI} \
    "torch==2.9.0+das.opt1.dtk2604"

if [ -n "${USE_FLAGTREE}" ]; then
  uv pip install --index ${FLAGOS_PYPI} \
    "flagtree==0.5.0+hcu3.0"
else
  uv pip install --index ${FLAGOS_PYPI} \
    "triton==3.3.0+das.opt1.dtk2604.torch290"
fi
