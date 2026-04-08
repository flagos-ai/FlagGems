---
title: 测试 Python 算子
weight: 20
---

<!--
# Testing Python Operators

*FlagGems* uses `pytest` for operator accuracy testing and performance benchmarking.
It  leverages Triton's `triton.testing.do_bench` for kernel-level performance evaluation.
-->
# 测试 Python 算子

*FlagGems* 使用 `pytest` 来驱动算子精度测试和性能基准测试。
项目使用 Triton 的 `triton.testing.do_bench` 来执行内核层级的性能评估。

<!--
## 1. Accuracy tests for operators

To run unit tests on a specific backend like CUDA:
-->
## 1. 算子精度测试

要在特定的后端硬件（如 CUDA）上运行测试：

```shell
pytest tests/test_${name}.py
```

<!--
The following command runs the tests on CPU:
-->
下面的命令执行在 CPU 上的精度测试：

```shell
pytest tests/test_${case}.py --ref cpu
```

<!--
## 2. Accuracy in the context of models
-->
## 2. 在具体模型下执行精度测试

```shell
pytest examples/${name}_test.py
```

<!--
## 3. Test operator performance

To test operator performance on CUDA:
-->
## 3. 测试算子的性能

在 CUDA 平台上测试算子的性能：

```shell
pytest benchmark/test_xx_perf.py -s
```

<!--
To benchmark the end-to-end performance for operators:
-->
下面的命令对算子执行端到端的性能基准测试：

```shell
pytest benchmark/test_xx_perf.py -s --mode cpu
```

<!--
## 4. Run tests with logging infomation
-->
## 4. 执行测试时输出日志信息

```bash
pytest program.py --log-cli-level debug
```

<!--
This is NOT recommended when conducting performance benchmarks.
-->
在执行性能基准测试时不建议启用日志输出。

<!--
## 5. The daily full-range unit tests
-->
## 5. 每日全量单元测试

(TBD)
