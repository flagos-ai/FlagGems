VENDOR=$1

export FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-${VENDOR}/simple"

case $VENDOR in
  ascend)
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

    uv pip install -e .
    uv pip install ".[test]"
    ;;

  enflame)
    uv pip install --index ${FLAGOS_PYPI} \
      "pyefml==1.9.10" \
      "torch==2.10.0+cpu" \
      "torchaudio==2.10.0+cpu" \
      "torchvision==0.25.0+cpu" \
      "torch-gcu==2.10.0+3.7.20260408" \
      "triton-gcu==3.6.0+1.0.20260521.cc.1.9.10" \
      "flash-attn==2.7.2+torch.2.9.1.gcu.3.4.20260323"

    # Replace triton with flagtree if requested
    # Currenly not working because it requires GLIBCXX_3.4.32
    # if [ -n "${USE_TRITON}" ]; then
    #   uv pip uninstall flagtree
    #   uv pip install --index ${FLAGOS_PYPI} \
    #     flagtree==0.5.0+enflame.gitadb592d5
    # fi
    uv pip install -e .
    uv pip install ".[test]"
    ;;

  hygon)
    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.9.0+das.opt1.dtk2604" \
        "flagtree==0.5.0+hcu3.0"

    # Replace flagtree with Triton if requested
    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip install --index ${FLAGOS_PYPI} \
        "triton==3.3.0+das.opt1.dtk2604.torch290"
    fi

    uv pip install -e .
    uv pip install ".[test]"
    ;;


  iluvatar)
    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.7.1+corex.4.4.0" \
        "torchaudio==2.7.1+corex.4.4.0" \
        "torchvision==0.22.1+corex.4.4.0" \
        "triton==3.1.0+corex.4.4.0"

    # Replace Triton by FlagTree when FlagTree doesn't coredump
    # if [ -z "${USE_TRITON}" ]; then
    #   uv pip uninstall triton
    #   uv pip install --index $FLAGOS_PYPI \
    #     "flagtree==0.5.1+iluvatar3.1"
    # fi

    uv pip install -e .
    uv pip install ".[test]"
    ;;

  kunlunxin)
    uv pip install --index ${FLAGOS_PYPI} \
        "apex==0.1" \
        "benchflow==1.0.0" \
        "colorama==0.4.6" \
        "flash_attn==2.4.2+7e2dd4d" \
        "hyperparameter==0.5.6" \
        "psutil==6.1.0" \
        "regex==2026.4.4" \
        "torch==2.9.0+cu129" \
        "torchaudio==2.9.0+cu129" \
        "torchvision==0.24.0+cu129" \
        "torch_plugin==0.1.0" \
        "torch_xray==2.0.4" \
        "xformers==0.0.29+1e7a8ec.d20260114" \
        "xmlir==1.0.0.1"

    # Override the default triton 3.5.0 pulled in
    uv pip install --index ${FLAGOS_PYPI} \
        "triton==3.0.0+a48aedef"

    # Install FlagTree if requested and ready
    # if [ -z ${USE_TRITON} ]; then
    #   uv pip uninstall flagtree
    #   uv pip install --index ${FLAGOS_PYPI} \
    #     "flagtree==3.0.0+0762702f"
    # fi
    uv pip install -e .
    uv pip install ".[test]"
    ;;

  metax)
    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.8.0+metax3.7.2.0" \
        "torchaudio==2.4.1+metax3.7.2.0" \
        "torchvision==0.15.1+metax3.7.2.0" \
        "flagtree==3.1.0+metax3.7.2.0" \
        "flash_attn==2.6.3+metax3.7.2.0torch2.8"

    # if [ -n "${USE_TRITON}" ]; then
    #   uv pip uninstall flagtree
    #   uv pip install --index ${FLAGOS_PYPI} \
    #     "triton==3.0.0+metax3.7.2.0"
    # fi

    uv pip install -e  .
    uv pip install ".[test]"
    ;;

  mthreads)
    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.9.0+musa.4.3.6" \
        "torch_musa==2.9.0" \
        "mkl==2024.0.0" \
        "triton==3.6.0+git89458660"

    # Replace flagtree with Triton if requested
    # if [ -n "${USE_TRITON}" ]; then
    #   uv pip uninstall flagtree
    #   uv pip uninstall triton
    #   uv pip install --index $FLAGOS_PYPI \
    #     "triton==3.6.0+git89458660"
    # else
    #   uv pip uninstall triton
    #   uv pip install --index $FLAGOS_PYPI \
    #     "flagtree==0.5.1+mthreads3.6"
    # fi

    uv pip install -e  .
    uv pip install ".[test]"
    ;;

  nvidia)
    # We need pytorch first for building C++ wrapped operators
    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.11.0+cu130" \
        "torchvision==0.26.0+cu130" \
        "triton==3.6.0"

    # The follow environments are for C++ wrapped operators
    # export CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
    # export CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=CUDA"
    uv pip install -e . --no-build-isolation
    uv pip install ".[test]"

    # We don't have flagtree for triton 3.6 yet
    # if [ -n "${USE_TRITON}" ]; then
    #   uv pip uninstall triton
    #   uv pip install --index ${FLAGOS_PYPI} \
    #     "flagtree==0.5.1+3.6"
    # fi
    ;;

  spacemit)
    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.8.0+spacemit.0" \
        "triton==3.6.0+spacemit.a5"

    uv pip install -e .
    uv pip install ".[test]"
    ;;

  sunrise)
    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.11.0+cpu" \
        "torchaudio==2.11.0+cpu" \
        "torchvision==0.26.0+cpu" \
        "torch-ptpu==0.2.1+gaf2c267.torch2.11" \
        "triton==3.4.0.5+git27a132da"

    # We try triton first at this stage
    # if [ -z "${USE_TRITON}" ]; then
    #   uv pip uninstall triton
    #   uv pip install --index ${FLAGOS_PYPI} \
    #     "flagtree-0.4.0+sunrise3.4"
    # fi
    uv pip install -e .
    uv pip install ".[test]"
    ;;

  thead)
    # TODO(Qiming): Investigate environment settings
    uv pip install -e .
    uv pip install ".[test]"
    ;;

  tsingmicro)
    uv pip install --index ${FLAGOS_PYPI} \
        "torch==2.7.0+cpu" \
        "torchvision==0.22.0+cpu" \
        "torchaudio==2.7.0+cpu" \
        "torch_txda==0.1.0+20260416.b8f53e8a" \
        "txops==0.1.0+20260508.60287151" \
        "flagtree==0.5.0+tsingmicro3.3"

    # Replace flagtree with Triton if requested
    if [ -n "${USE_TRITON}" ]; then
      uv pip uninstall flagtree
      uv pip install --index ${FLAGOS_PYPI} \
        "triton==3.3.0+git2e9c1195"
    fi

    uv pip install -e .
    uv pip install ".[test]"
    ;;

  *)
    echo "Unknown backend ${VENDOR}"
    ;;
esac
