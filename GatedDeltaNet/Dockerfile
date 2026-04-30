# Use PyTorch with CUDA and cuDNN
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Install git, python3-packaging and check Python and pip versions
RUN apt-get update && apt-get install -y git python3-packaging \
    && python -c "import sys; print(sys.version)" \
    && pip --version

Run pip install ninja
RUN pip install packaging torch>=2.1.0dev lightning==2.1.2 lightning[app]
RUN pip install jsonargparse[signatures] tokenizers sentencepiece wandb lightning[data] torchmetrics
RUN pip install tensorboard zstandard pandas pyarrow huggingface_hub
RUN pip install -U flash-attn --no-build-isolation
RUN git clone https://github.com/Dao-AILab/flash-attention
WORKDIR flash-attention
WORKDIR csrc/rotary
RUN pip install .
WORKDIR ../layer_norm
RUN pip install .
WORKDIR ../xentropy
RUN pip install .
RUN pip install --no-cache-dir datasets==2.20.0 triton==2.3.0 xformers tomli>=1.1.0 causal-conv1d>=1.2.0 mamba-ssm transformers
Run pip install einops
Run pip install braceexpand
Run pip install smart_open
Run pip install opt_einsum
Run pip install cbor2
Run pip install isort
Run pip install pytest
Run pip install mypy
Run pip install mosaicml-streaming
Run pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
RUN pip uninstall torchdata -y
RUN pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly
RUN pip install --no-cache-dir lm-eval==0.4.1
