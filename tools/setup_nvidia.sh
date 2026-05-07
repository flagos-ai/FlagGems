#!/bin/bash

uv pip install -e .
uv pip install ".[nvidia,test]"

uv pip install --index ${FLAGOS_PYPI} \
    "torch==2.9.0+cu128" \
    "torchvision==0.24.0+cu128" \
    "torchaudio==2.9.0+cu128"

if [ -n "${USE_TRITON}" ]; then
  uv pip uninstall flagtree
  uv pip install --index ${FLAGOS_PYPI} \
    "triton==3.5"
fi
