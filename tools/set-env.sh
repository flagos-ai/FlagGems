VENDOR=$1
echo "Setting up environment variable for vendor $VENDOR"

case $VENDOR in
  ascend)
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
    ;;
  hygon)
    source /opt/dtk-26.04/env.sh
    ;;
  iluvatar)
    export LD_LIBRARY_PATH=/usr/local/corex/lib:$LD_LIBRARY_PATH
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
  mthreads)
    export MUSA_HOME=/usr/local/musa
    export PATH=$MUSA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
    if [ -z "${USE_TRITON}" ]; then
      SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.10/site-packages
      export LD_LIBRARY_PATH=${SITE_PACKAGES}/triton/_C:$LD_LIBRARY_PATH
    fi
  tsingmicro)
    export TX8_DEPS_ROOT=/opt/tsingmicro/tx8_deps
    export LLVM_SYSPATH=/opt/tsingmicro/llvm21
    export LLVM_BINARY_DIR=${LLVM_SYSPATH}/bin
    export PYTHONPATH=${LLVM_SYSPATH}/python_packages/mlir_core:$PYTHONPATH
    export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:/usr/local/kuiper/lib:$LD_LIBRARY_PATH
    if [ -n "${USE_TRITON}" ]; then
      SITE_PACKAGES=$VIRTUAL_ENV/lib/python3.10/site-packages
      export PYTHONPATH=$SITE_PACKAGES/triton/backends/tsingmicro/llvm/python_packages/mlir_core
    fi
esac
