```
#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}"

echo "===== [1] Clean build cache ====="
rm -rf ./build ./dist ./*.egg-info ./.skbuild
pip uninstall -y flag-gems || true

echo "===== [2] Install FlagGems (debug mode, triton kernel hot load) ====="
pip install --no-build-isolation -e .

echo "===== [3] Import smoke test ====="
python -c "import flag_gems; print('[OK] FlagGems loaded, path =', flag_gems.__file__)"

echo "===== [4] Run mm family unit test ====="
python -m pytest tests/test_mm.py -v
python -m pytest tests/test_addmm.py -v
python -m pytest tests/test_bmm.py -v

echo "===== [5] Run conv2d & glu smoke ====="
python -m pytest tests/test_conv2d.py -v
python -m pytest tests/test_glu.py -v

echo "===== All compile & smoke test passed! Ready for FlagPerf benchmark ====="
```

```
chmod +x rebuild.sh
./rebuild.sh
```
