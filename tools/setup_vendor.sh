VENDOR=$1

export FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-${VENDOR}/simple"

case $VENDOR in
  ascend)
    uv pip install -e .
    uv pip install ".[ascend,test]"

    uv pip install --index ${FLAGOS_PYPI} \
        "flagtree==0.5.0+ascend3.2" \
        "torch==2.9.0+cpu" \
        "torch-npu==2.9.0"

    # Replace flagtree with Triton if requested
    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip install --index ${FLAGOS_PYPI} \
        triton_ascend==3.2.0
    fi
    ;;

  hygon)
    uv pip install -e .
    uv pip install ".[hygon]"
    uv pip install ".[test]"

    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.9.0+das.opt1.dtk2604" \
        "flagtree==0.5.0+hcu3.0"

    # Replace flagtree with Triton if requested
    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip install --index ${FLAGOS_PYPI} \
        "triton==3.3.0+das.opt1.dtk2604.torch290"
    fi
    ;;


  iluvatar)
    uv pip install -e .
    uv pip install ".[iluvatar, test]"

    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.7.1+corex.4.4.0" \
        "torchaudio==2.7.1+corex.4.4.0" \
        "torchvision==0.22.1+corex.4.4.0" \
        "flagtree==0.5.1+iluvatar3.1"

    # Replace flagtree by Triton if requested
    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip install --index $FLAGOS_PYPI \
        "triton==3.1.0+corex.4.4.0"
    fi
    ;;

  kunlunxin)
    uv pip install -e .
    uv pip install ".[kunlunxin,test]"

    uv pip install --index ${FLAGOS_PYPI} \
        nvidia-cublas-cu11==11.11.3.6 \
        nvidia-cuda-cupti-cu11==11.8.87 \
        nvidia-cuda-nvrtc-cu11==11.8.89 \
        nvidia-cuda-runtime-cu11==11.8.89 \
        nvidia-cudnn-cu11==9.1.0.70 \
        nvidia-cufft-cu11==10.9.0.58 \
        nvidia-curand-cu11==10.3.0.86 \
        nvidia-cusolver-cu11==11.4.1.48 \
        nvidia-cusparse-cu11==11.7.5.86 \
        nvidia-nccl-cu11==2.21.5 \
        nvidia-nvtx-cu11==11.8.86

    uv pip install --index ${FLAGOS_PYPI} \
        "benchflow==1.0.0" \
        "hyperparameter==0.5.6" \
        "torch==2.5.1+cu118" \
        "torchaudio==2.5.1+cu118" \
        "torchvision==0.20.1+cu118" \
        "torch_klx==0.1.0" \
        "torch_xray==0.2.1" \
        "psutil==6.1.0" \
        "regex==2026.4.4" \
        "xmlir==1.0.0.1"

    # Install Triton if requested
    if [ -n ${USE_TRITON} ]; then
      uv pip uninstall flagtree
      uv pip install --index ${FLAGOS_PYPI} \
        "triton=3.0.0+0762702f"
    fi
    ;;

  metax)
    uv pip install -e  .
    uv pip install ".[metax]"
    uv pip install ".[test]"

    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.8.0+metax3.5.3.9" \
        "torchaudio==2.4.1+metax3.5.3.9" \
        "torchvision==0.15.1+metax3.5.3.9" \
        "flagtree==3.1.0+metax3.5.3.9"

    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip install --index ${FLAGOS_PYPI} \
        "triton==3.0.0+metax3.5.3.9"
    fi
    ;;

  mthreads)
    uv pip install -e .
    uv pip install ".[mthreads,test]"

    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.7.1+musa.4.0.0" \
        "torch_musa==2.7.1" \
        "numpy==1.26.4" \
        "mkl==2024.0.0"

    # Replace flagtree with Triton if requested
    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip uninstall triton
      uv pip install --index $FLAGOS_PYPI \
        "triton==3.1.0+musa1.4.6"
    else
      uv pip uninstall triton
      uv pip install --index $FLAGOS_PYPI \
        "flagtree==0.5.0+mthreads3.1"
    fi
    ;;

  nvidia)
    uv pip install -e .
    uv pip install ".[nvidia,test]"

    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.9.0+cu128" \
        "torchvision==0.24.0+cu128" \
        "torchaudio==2.9.0+cu128" \
        "flagtree==0.5.0+3.5"

    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip install --index ${FLAGOS_PYPI} \
        "triton==3.5"
    fi
    ;;

  thead)
    # TODO(Qiming): Investigate environment settings
    uv pip install -e .
    uv pip install ".[thead,test]"
    ;;

  tsingmicro)
    uv pip install -e .
    uv pip install ".[tsingmicro]"
    uv pip install ".[test]"

    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.7.0+cpu" \
        "torchvision==0.22.0" \
        "torchaudio==2.7.0" \
        "torch_txda==0.1.0+20260310.294fc4a6" \
        "txops==0.1.0+20260225.5cc33e4e" \
        "flagtree==0.5.0+tsingmicro3.3"

    # Replace flagtree with Triton if requested
    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip install --index ${FLAGOS_PYPI} \
        "triton==3.3.0+gitfe2a28fa"
    fi
    ;;

  *)
    echo "Unknown backend ${VENDOR}"
    ;;
esac
