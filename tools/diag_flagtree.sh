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

echo "--- all triton installations on this machine ---"
python - <<'PY' 2>&1 || true
import hashlib
import glob
import os
import site
import sys
from importlib import metadata

def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except OSError:
        return "<unreadable>"

def describe(pkg):
    init = os.path.join(pkg, "__init__.py")
    xpu = os.path.join(pkg, "backends", "xpu", "compiler.py")
    fb = os.path.join(pkg, "FLAGTREE_BACKEND")
    bits = [pkg]
    if os.path.isfile(xpu):
        bits.append("xpu=" + sha(xpu))
    if os.path.isfile(fb):
        bits.append("backend=" + open(fb).read().strip())
    return "  ".join(bits)

found = {}

def add(pkg):
    pkg = os.path.realpath(pkg)
    if os.path.isfile(os.path.join(pkg, "__init__.py")):
        found[pkg] = None

# 1) every sys.path entry is what `import triton` actually searches
for entry in sys.path:
    if entry:
        add(os.path.join(entry, "triton"))

# 2) site-packages / user site of the running interpreter
for d in site.getsitepackages():
    add(os.path.join(d, "triton"))
add(os.path.join(site.getusersitepackages(), "triton"))
for pattern in (
    "/usr/lib/python*/site-packages",
    "/usr/lib/python*/dist-packages",
    "/usr/local/lib/python*/site-packages",
    "/usr/local/lib/python*/dist-packages",
):
    for d in glob.glob(pattern):
        add(os.path.join(d, "triton"))

# 3) baked trees outside the venv (e.g. /opt/triton injected via PYTHONPATH,
#    or an image-baked /opt/flagtree): bounded walk for any triton/__init__.py
def walk(root, depth=0):
    if depth > 5 or not os.path.isdir(root):
        return
    try:
        entries = os.listdir(root)
    except OSError:
        return
    if os.path.isfile(os.path.join(root, "triton", "__init__.py")):
        add(os.path.join(root, "triton"))
    for name in entries:
        if name in (".git", "__pycache__", "node_modules"):
            continue
        child = os.path.join(root, name)
        if os.path.isdir(child):
            walk(child, depth + 1)

for root in ("/opt", "/usr/local", "/flagos"):
    walk(root)

print("triton packages (dirs containing __init__.py):")
for pkg in sorted(found):
    print("  " + describe(pkg))
if not found:
    print("  (none found)")

print("pip distributions named triton:")
triton_dists = 0
for dist in metadata.distributions():
    name = (dist.metadata.get("Name") or "").replace("_", "-").lower()
    if name == "triton":
        triton_dists += 1
        try:
            loc = str(dist.locate_file(""))
        except Exception:
            loc = "<unknown>"
        print(f"  triton {dist.version}  {loc}")
if triton_dists == 0:
    print("  (none)")

print("flagtree distributions (and whether they bundle triton/):")
flagtree_dists = 0
for dist in metadata.distributions():
    name = (dist.metadata.get("Name") or "").replace("_", "-").lower()
    if name == "flagtree":
        flagtree_dists += 1
        ships = any(str(f).startswith("triton/") for f in (dist.files or []))
        try:
            loc = str(dist.locate_file(""))
        except Exception:
            loc = "<unknown>"
        print(f"  flagtree {dist.version}  ships_triton={ships}  {loc}")
if flagtree_dists == 0:
    print("  (none)")
PY

echo "==================== diagnose done ===================="
