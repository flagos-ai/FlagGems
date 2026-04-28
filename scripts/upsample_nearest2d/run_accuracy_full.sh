#!/usr/bin/env bash
set -euo pipefail

pytest tests/test_upsample_nearest2d.py -s

if [ -f tests/test_upsample_nearest2d_backward.py ]; then
  pytest tests/test_upsample_nearest2d_backward.py -s
fi

pytest tests/test_upsample_nearest1d.py -s
pytest tests/test_upsample_nearest3d.py -s

