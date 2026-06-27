BACKEND=$1
echo "Setting up environment variable for backend $BACKEND"

# Vendor env scripts append to these variables without guarding against unset.
# Default them here so callers with `set -u` (e.g. setup.sh) don't fail.
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
export C_INCLUDE_PATH="${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${LIBRARY_PATH:-}"

case $BACKEND in
  ascend-cann850|ascend-cann900)
    # This script is provided by the Huawei Ascend CANN toolkit installation.
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
    if [ -f /usr/local/Ascend/toolbox/set_env.sh ]; then
      source /usr/local/Ascend/toolbox/set_env.sh
    fi

    # TODO: Check if this is necessary
    # export TRITON_ALL_BLOCKS_PARALLEL=1
    ;;
  cambricon)
    export PATH=/usr/local/neuware/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH
    ;;
  enflame)
    # gcc-toolset-14 provides GLIBCXX_3.4.32+ required by some packages
    if [ -d /opt/OpenCloudOS/gcc-toolset-14/root/usr/lib64 ]; then
      export LD_LIBRARY_PATH=/opt/OpenCloudOS/gcc-toolset-14/root/usr/lib64:$LD_LIBRARY_PATH
    fi
    ;;
  hygon)
    source /opt/dtk-26.04/env.sh
    echo "PATH=$PATH"
    ;;
  iluvatar)
    export COREX_ROOT=/usr/local/corex
    export PATH=$COREX_ROOT/bin:$PATH
    export LD_LIBRARY_PATH=$COREX_ROOT/lib:$LD_LIBRARY_PATH
    ;;
  kunlunxin)
    export LD_LIBRARY_PATH=/xcudart/lib:/usr/local/cuda/lib64
    ;;
  metax)
    export MACA_PATH=/opt/maca
    export LD_LIBRARY_PATH=$MACA_PATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH
    if [ -z "${USE_TRITON}" ]; then
      SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.12/site-packages
      export LD_LIBRARY_PATH=${SITE_PACKAGES}/triton/backends/metax/lib:$LD_LIBRARY_PATH
    fi
    ;;
  nvidia)
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ;;
  mthreads)
    export MUSA_HOME=/usr/local/musa
    export PATH=$MUSA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
    if [ -z "${USE_TRITON}" ]; then
      SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.10/site-packages
      export LD_LIBRARY_PATH=${SITE_PACKAGES}/triton/_C:$LD_LIBRARY_PATH
    fi
    ;;
  thead)
    # The envsetup.sh is provided by the PPU SDK
    source /usr/local/PPU_SDK/envsetup.sh
    ;;
  tsingmicro)
    export TX8_DEPS_ROOT=/opt/tx8_deps
    export LLVM_SYSPATH=/opt/llvm
    export LLVM_BINARY_DIR=${LLVM_SYSPATH}/bin
    export PYTHONPATH=${LLVM_SYSPATH}/python_packages/mlir_core
    export LD_LIBRARY_PATH=/usr/local/kuiper/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/kuiper/tsm8-profiler/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${TX8_DEPS_ROOT}/lib:${LD_LIBRARY_PATH}

    # if [ -n "${USE_TRITON}" ]; then
    #   export PYTHONPATH=$SITE_PACKAGES/triton/backends/tsingmicro/llvm/python_packages/mlir_core
    # fi
    ;;
esac
