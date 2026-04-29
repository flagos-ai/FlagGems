#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH

uv pip install -e .
uv pip install ".[iluvatar]"
uv pip install ".[test]"

uv pip install --index ${FLAGOS_PYPI} \
    "torch==2.7.1+corex.4.4.0" \
    "torchaudio==2.7.1+corex.4.4.0" \
    "torchvision==0.22.1+corex.4.4.0" \
    "flagtree==0.5.0+iluvatar3.1"

# Replace flagtree by Triton if requested
if [ -n "${USE_TRITON}" ]; then
  uv pip uninstall flagtree
  uv pip install --index $FLAGOS_PYPI \
    "triton==3.1.0+corex.4.4.0"
fi
