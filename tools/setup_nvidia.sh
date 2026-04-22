#!/bin/bash

uv pip install -e . .[nvidia,test]

uv pip install --index ${FLAGOS_PYPI} \
    "torch==2.9.0+cu128" \
    "torchvision==0.24.0+cu128" \
    "torchaudio==2.9.0+cu128"

if [ -n "${USE_FLAGTREE}" ]; then
  uv pip uninstall triton
  uv pip install --index ${FLAGOS_PYPI} \
    "flagtree==0.5.0+3.5"
else
  uv pip install --index ${FLAGOS_PYPI} \
    "triton==3.5"
fi
