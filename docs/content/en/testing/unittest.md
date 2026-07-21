---
title: Testing Python Operators
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
pytest tests/test_foo.py --ref cpu
```

## 2. Accuracy in the context of models

```shell
pytest examples/${name}_test.py
```

## 3. Test operator performance

To test CUDA performance

```shell
pytest benchmark/test_foo.py -s
```

To benchmark the end-to-end performance for operators:

```shell
pytest benchmark/test_foo.py -s --ref cpu
```
