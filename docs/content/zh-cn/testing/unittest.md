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

# Python 算子测试指南

*FlagGems* 统一采用 `pytest` 作为精度测试与性能评测的驱动引擎，并利用 Triton 的 `triton.testing.do_bench` 进行微架构级底层 Kernel 性能度量。

## 1. 算子单元精度测试

在当前目标计算设备（如 CUDA 或 NPU）上运行单个算子的精度测试套件：

```shell
pytest tests/test_${op_name}.py
```

若需强制以 CPU 计算结果作为基准参考进行双精度比对：

```shell
pytest tests/test_${op_name}.py --ref cpu
```

## 2. 模型上下文精度对齐验证

运行实际模型端到端前向/反向输出一致性验证：

```shell
pytest examples/${model_name}_test.py
```

## 3. 算子性能基准测试

直接运行性能评测脚本并打印详细耗时指标：

```shell
pytest benchmark/test_${op_name}_perf.py -s
```

## 4. 测试套件专用命令行选项

除 `pytest` 原生参数（如 `-k` 关键字筛选、`-m` 标记筛选、`-x` 遇错即停）外，*FlagGems* 在 `tests/conftest.py` 中注册了如下专有测试增强参数：

| 选项名称 | 可选值 | 说明 |
| :------- | :----- | :--- |
| `--ref` | `<device>`（默认）、`cpu` | 指定生成真值（Ground Truth）的参考计算设备 |
| `--quick` | 开关标志 | 自动缩减测试用的 Shape 与数据类型组合，适用于快速 Smoke Test 验证 |
| `--record` | `none`（默认）、`log`、`json` | 将测试运行的明细元数据持久化记录为日志或 JSON 报表 |
| `--output` | 路径字符串 | 使用 `--record json` 时的落盘报表文件名（默认 `accuracy_result.json`） |
| `--collect-marks` | 路径字符串 | 仅静态收集测试用例标记（Markers）并导出为 YAML 配置文件，不实际执行计算 |

示例：使用 CPU 作为参考真值对 `softmax` 算子进行快速冒烟测试，并将结果保存为 JSON：

```shell
pytest tests/test_softmax.py --ref cpu --quick --record json --output softmax_result.json
```

### 基于 Pytest Marker 灵活筛选

测试套件为每个算子精度用例打上了同名 Marker（例如 `@pytest.mark.fliplr`）。这允许开发者无需记忆文件名即可按算子名灵活过滤：

```shell
pytest tests/ -m "fliplr or rms_norm"
```

### 测试报告生成与代码覆盖率统计

*FlagGems* 的持续集成流水线结合了 [`pytest-md-report`](https://pypi.org/project/pytest-md-report/) 生成 Markdown 摘要，并使用 [`coverage`](https://coverage.readthedocs.io/) 统计分支测试覆盖率：

```shell
coverage run -m pytest -s tests/test_foo.py \
    --md-report --md-report-verbose=1 --md-report-output=summary.md
coverage html
```

更多覆盖率指标与门禁配置，请参阅[单元测试覆盖率](coverage.md)。
