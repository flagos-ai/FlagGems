#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install -e ".[dev]" || python -m pip install -e .

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

