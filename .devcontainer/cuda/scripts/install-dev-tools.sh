#!/usr/bin/env bash
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
#pip install -r "$REPO_ROOT/requirements/requirements_nvidia.txt"

git submodule sync
git submodule update --init --recursive

CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DCMAKE_BUILD_TYPE=Release" \
pip install -v --no-build-isolation -e .

