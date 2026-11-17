#!/usr/bin/env bash
# Diagnose which flagtree/triton the CI container actually loads.
# Prints the real import path, version, PYTHONPATH and whether the prebuilt
# /opt/flagtree tree exists, so the "unpack reached lowering without a bufPtr"
# failure can be attributed to the correct compiler installation.
set -uo pipefail

echo "==================== flagtree/triton diagnose ===================="

echo "--- python executable ---"
which python python3 2>&1 || true

echo "--- import triton: real file + version ---"
python -c "import triton; print('TRITON_FILE =', triton.__file__); print('TRITON_VER  =', getattr(triton, '__version__', 'N/A'))" 2>&1 || true

echo "--- import flagtree: version + file ---"
python -c "import flagtree; print('FLAGTREE_VER  =', getattr(flagtree, '__version__', 'N/A')); print('FLAGTREE_FILE =', flagtree.__file__)" 2>&1 || true

echo "--- pip show flagtree ---"
pip show flagtree 2>&1 | grep -E '^(Name|Version|Location):' || true

echo "--- sys.path entries mentioning flagtree/triton ---"
python -c "import sys; [print(p) for p in sys.path if 'flagtree' in p.lower() or 'triton' in p.lower()]" 2>&1 || true

echo "--- PYTHONPATH ---"
echo "PYTHONPATH=[${PYTHONPATH:-<empty>}]"

echo "--- /opt/flagtree prebuilt tree ---"
ls -d /opt/flagtree 2>&1 || true
ls -la /opt/flagtree/triton/backends/xpu/compiler.py 2>&1 || true

echo "--- site-packages flagtree dist-info ---"
python - <<'PY' 2>&1 || true
import site, glob, os
for d in site.getsitepackages():
    for m in sorted(glob.glob(os.path.join(d, 'flagtree-*.dist-info'))):
        print('DISTINFO:', m)
PY

echo "==================== diagnose done ===================="
