sh ~/clear-triton-cache.sh
export FLAGTREE_AABS=0
#python3 -m pytest -s test_mm.py -m mm1 --level core --dtypes bfloat16 -x --mode=kernel
python3 -m pytest -s test_mm.py -m mm1 --level core --dtypes float32 -x --mode=kernel

