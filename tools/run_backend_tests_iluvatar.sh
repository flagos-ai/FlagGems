#!/bin/bash

VENDOR=${1:?"Usage: bash tools/run_backend_tests_iluvatar.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

echo "Running FlagGems tests with GEMS_VENDOR=$VENDOR"

export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

pip install -U pip
pip install uv
uv venv
source .venv/bin/activate
uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0
uv pip install -e .[iluvatar,test]

pytest -s tests
