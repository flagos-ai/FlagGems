#!/usr/bin/env bash
set -euo pipefail

pytest benchmark/test_upsample_nearest2d.py -s --record log

if [ -f benchmark/test_upsample_nearest2d_backward.py ]; then
  pytest benchmark/test_upsample_nearest2d_backward.py -s --record log
fi

