#!/bin/bash

# uv pip install --index $FLAGOS_PYPI \
#        "torch==2.7.1+corex.4.4.0" \
#        "torchaudio==2.7.1+corex.4.4.0" \
#        "torchvision==0.22.1+corex.4.4.0" \
#        "triton==3.1.0+corex.4.4.0"
uv pip install -e .[iluvatar,test]

export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH
