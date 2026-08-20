---
title: 测试 Python 算子
weight: 20
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


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
pytest benchmark/test_foo.py -s
```

<!--
To benchmark the end-to-end performance for operators:
-->
下面的命令对算子执行端到端的性能基准测试：

```shell
pytest benchmark/test_foo.py -s --ref cpu
```
