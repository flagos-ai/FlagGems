#!/bin/bash

# Environment setting for DTK-26.04
source /opt/dtk-26.04/env.sh

uv pip install -e .
uv pip install ".[hygon]"
uv pip install ".[test]"

uv pip install --index ${FLAGOS_PYPI} \
    "torch==2.9.0+das.opt1.dtk2604" \
    "flagtree==0.5.0+hcu3.0",

# Replace flagtree with Triton if requested
if [ -n "${USE_TRITON}" ]; then
  uv pip uninstall flagtree
  uv pip install --index ${FLAGOS_PYPI} \
    "triton==3.3.0+das.opt1.dtk2604.torch290"
fi
