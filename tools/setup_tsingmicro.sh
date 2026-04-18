#!/bin/bash

# Tsingmicro dependencies
export TX8_DEPS_ROOT=/opt/tsingmicro/tx8_deps
export LLVM_SYSPATH=/opt/tsingmicro/llvm21
export LLVM_BINARY_DIR=${LLVM_SYSPATH}/bin
export PYTHONPATH=${LLVM_SYSPATH}/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:/usr/local/kuiper/lib:$LD_LIBRARY_PATH


uv pip install -e .[tsingmicro,test]

# uv pip install --index ${FLAGOS_PYPI} \
#   torch_txda==0.1.0+20260310.294fc4a6 \
#    txops==0.1.0+20260225.5cc33e4e
# uv pip install triton==3.3.0+gitfe2a28fa --index $FLAGOS_PYPI

# Use flagtree
uv pip uninstall triton
uv pip install --index ${FLAGOS_PYPI} \
    flagtree==0.5.0+tsingmicro3.3

# The following is needed when using `triton==3.3.0+gitfe2a28fa` rather than `flagtree`
# SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.10/site-packages
# export PYTHONPATH=$SITE_PACKAGES/triton/backends/tsingmicro/llvm/python_packages/mlir_core

# NOTE: The following setting may be needed if there are exceptions related to txops.
# export LD_LIBRARY_PATH=$SITE_PACKAGES/txops/lib:$LD_LIBRARY_PATH
