---
title: Testing Python Operators
weight: 20
---

# Testing Python Operators

*FlagGems* uses `pytest` for operator accuracy testing and performance benchmarking.
It  leverages Triton's `triton.testing.do_bench` for kernel-level performance evaluation.

## 1. Accuracy tests for operators

To run unit tests on a specific backend like CUDA:

```shell
pytest tests/test_${name}.py
```

The following command runs the tests on CPU:

```shell
pytest tests/test_${case}.py --ref cpu
```

## 2. Accuracy in the context of models

```shell
pytest examples/${name}_test.py
```

## 3. Test operator performance

To test CUDA performance

```shell
pytest benchmark/test_xx_perf.py -s
```

To benchmark the end-to-end performance for operators:

```shell
pytest benchmark/test_xx_perf.py -s --mode cpu
```

## 4. Run tests with logging infomation

```bash
pytest program.py --log-cli-level debug
```

This is NOT recommended when conducting performance benchmarks.

## 5. The daily full-range unit tests

(TBD)
