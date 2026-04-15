#!/bin/bash
# test_backends.sh <VENDOR> <PR_ID>

echo $CHANGED_FILES

SUPPORTED_VENDORS=(
  "ascend"
  "hygon"
  "iluvatar"
  "kunlunxin"
  "metax"
  "mthreads"
  "nvidia"
  "thead"
  "tsingmicro"
)

valid_vendor() {
  needle=$1
  for item in "${SUPPORTED_VENDORS[@]}" ; do
    [ $item == "$needle" ] && return 0
  done
  return 1
}

[ "$#" -eq 2 ] || { echo "Please specify <VENDOR> and <PR_ID>"; exit 1; }
VENDOR=${1}
valid_vendor $VENDOR
[ "$?" == 0 ] || { echo "Invalid vendor ${VENDOR} specified" ; exit 1; }
PR_ID=${2}

echo "Running FlagGems tests with VENDOR=${VENDOR}"

# 1. Set virtual environment
#    This is specific to the CI runners
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

uv venv
source .venv/bin/activate

# 2. Install
export FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-${VENDOR}/simple"
uv pip install \
  "setuptools==79.0.1" \
  "scikit-build-core==0.12.2" \
  "pybind11==3.0.3" \
  "cmake>=3.20,<4" \
  "ninja==1.13.0"

## Vendor-specific installation steps
source tools/setup_${VENDOR}.sh
[ "$?" == 0 ] || { echo "Failed to setup FlagGems" ; exit 1; }

# 3. Run tests
# TODO(Qiming): Handle CHANGED_FILES and other parameters
# TODO(Qiming): Run performance tests as well
# TODO(Qiming): Merge the following logic with run_tests.py
tools/test-op.sh $PR_ID
