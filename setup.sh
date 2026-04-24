#!/bin/bash

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

# TODO: Add thead PPU
declare -A PYTHON_SUPPORTED=(
  ["ascend"]="3.11"
  ["hygon"]="3.10"
  ["iluvatar"]="3.10"
  ["kunlunxin"]="3.10"
  ["metax"]="3.12"
  ["mthreads"]="3.10"
  ["nvidia"]="3.12"
  ["tsingmicro"]="3.10"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

valid_vendor() {
  needle=$1
  for item in "${SUPPORTED_VENDORS[@]}" ; do
    [ $item == "$needle" ] && return 0
  done
  return 1
}

# Validate argument count
[ "$#" -eq 1 ] || { echo "Please specify <VENDOR>"; exit 1; }

# Validate vendor name
VENDOR=${1}
valid_vendor $VENDOR
if [ "$?" != 0 ]; then
    echo "Invalid vendor '${VENDOR}' specified ..."
    echo "Please specify one of: ${SUPPORTED_VENDORS[@]}"
    exit 1
fi

# Validate Python version
printf "Checking Python version ... "
python_version=$(python --version 2>/dev/null | awk '{print $NF}')
expected_version=${PYTHON_SUPPORTED[$VENDOR]}
if [[ "$python_version" == *"$expected_version"* ]]; then
  printf "${python_version} $GREEN[OK]$NC\n"
else
  printf "${python_version}, expecting '${expected_version}.*' $RED[FAILED]$NC"
  exit 1
fi

# Validate uv install
printf "Checking uv ... "
uv_version=$(uv --version 2>/dev/null | awk '{print $NF}')
if [ "$?" == 0 ];  then
  printf "${uv_version} ${GREEN}[OK]${NC}\n"
else
  printf "${RED}NOT FOUND${NC}\n"
  # Install uv and upgrade pip if necessary
  printf "Installing/upgrading pip and uv ... "
  pip install -U pip uv || exit 1;
fi

printf "Detecting pyenv ... "
pyenv_version=$(pyenv --version 2>/dev/null | awk '{print $NF}')
if [ "$?" != 0 ]; then
  # pyenv not installed
  printf "NOT FOUND $GREEN[OK]$NC\n"
else
  printf "${pyenv_version} $GREEN[OK]$NC\n"

  # Initialize pyenv virtual environment
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init - bash)"
fi

# Start installation
printf "Installing FlagGems for ${VENDOR}\n"

printf "Creating virtual environment ... "
uv venv -q
if [ "$?" != 0 ]; then
  printf "$RED{FAILED]$NC\n"
  exit 1
else
  printf "$RED{OK]$NC\n"
  source .venv/bin/activate
fi

# Install
export FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-${VENDOR}/simple"
printf "Install build tools ... "
uv pip install -q
  "setuptools>=64.0" \
  "scikit-build-core==0.12.2" \
  "pybind11==3.0.3" \
  "cmake>=3.20,<4" \
  "ninja==1.13.0"

# export USE_TRITON=0

## Vendor-specific installation steps
source tools/setup_${VENDOR}.sh

[ "$?" == 0 ] || { echo "Failed to setup FlagGems" ; exit 1; }
