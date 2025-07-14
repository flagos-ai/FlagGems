# Pytest in FlagGems

FlagGems uses `pytest` for operator accuracy and performance testing, and further leverages Triton's `triton.testing.do_bench` for kernel-level performance evaluation.

### 1. Test Operator Accuracy

- Run reference on specific backend like cuda

```bash
cd tests
pytest test_${op_name}_ops.py
```

- Run reference on cpu

```bash
cd tests
pytest test_${op_name}_ops.py --ref cpu
```

### 2. Test Model Accuracy

```bash
cd examples
pytest model_${model_name}_test.py
```

### 3. Test Operator Performance

- Test CUDA performance
  ```bash
  cd benchmark
  pytest test_xx_perf.py -s
  ```
- Test end-to-end performance
  ```bash
  cd benchmark
  pytest test_xx_perf.py -s --mode cpu
  ```

### 4. Run tests with logging infomation

```bash
pytest program.py --log-cli-level debug
```

And this is NOT recommended in performance testing.
