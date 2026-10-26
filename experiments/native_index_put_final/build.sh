#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXT_DIR="${ROOT}/experiments/native_ext"

cd "${EXT_DIR}"
MAX_JOBS="${MAX_JOBS:-2}" python setup.py build_ext --inplace
python -c 'import native_index_put_ext; print("Native MUSA extension import: PASS")'
