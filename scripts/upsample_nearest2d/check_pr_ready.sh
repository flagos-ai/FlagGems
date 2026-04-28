#!/usr/bin/env bash
set -euo pipefail

python -m ruff check src/flag_gems/ops/upsample_nearest2d.py tests/test_upsample_nearest2d.py benchmark/test_upsample_nearest2d.py

if [ -f src/flag_gems/ops/upsample_nearest2d_backward.py ]; then
  python -m ruff check src/flag_gems/ops/upsample_nearest2d_backward.py
fi

if [ -f tests/test_upsample_nearest2d_backward.py ]; then
  python -m ruff check tests/test_upsample_nearest2d_backward.py
fi

if [ -f benchmark/test_upsample_nearest2d_backward.py ]; then
  python -m ruff check benchmark/test_upsample_nearest2d_backward.py
fi

bash scripts/upsample_nearest2d/run_accuracy_quick.sh

