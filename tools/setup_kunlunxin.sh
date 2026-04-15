#!/bin/bash

export LD_LIBRARY_PATH=/xcudart/lib:/usr/local/cuda/lib64

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
    nvidia-nvtx-cu11==11.8.86 \

uv pip install --index ${FLAGOS_PYPI} \
    "benchflow==1.0.0" \
    "hydra==0.1.0" \
    "torch==2.5.1+cu118" \
    "torchaudio==2.5.1+cu118" \
    "torchvision==0.20.1+cu118" \
    "torch_klx==0.1.0" \
    "torch_xray==0.2.1" \
    "triton==3.0.0+0762702f" \
    "xmlir==1.0.0.1"

# Remove some wheels that might be problematic
uv pip uninstall pytest-repeat pytest-timeout

uv pip install -e .[kunlunxin,test]
