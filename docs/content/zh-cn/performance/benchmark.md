---
title: 性能基准测试
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

# 算子性能基准测试开发指南

在为 *FlagGems* 贡献新算子或优化现有实现时，建议遵循以下标准步骤为其编写性能评测用例。本规范同时适用于 Python 算子与 C++ 封装算子。

{{% steps %}}

1. **定位并选择对应的评测脚本**

   根据待测算子的数学运算特征，在仓库的 `benchmark/` 目录下匹配对应的测试文件：
   - 规约类算子（Reduction）：添加到 `test_reduction_perf.py`；
   - 张量构造类算子（Tensor Constructor）：添加到 `test_tensor_constructor_perf.py`；
   - 逐元素一元/二元算子：分别添加到 `test_unary_pointwise_perf.py` 或 `test_binary_pointwise_perf.py`；
   - 其他特种或复合算子：可添加到 `test_special_perf.py`，或按功能新建独立的评测脚本。

2. **评估复用现有的 Benchmark 基类**

   选定文件后，查阅该文件中继承自 `Benchmark` 的现有评测类，评估是否可直接复用：
   - **指标采集维度（Metric Collection）**：耗时（Latency/ms）、显存吞吐量（GB/s）或算力利用率（TFLOPS）是否契合；
   - **输入生成逻辑（Input Generator）**：现有的 `input_generator` 或 `input_fn` 是否支持生成目标算子所需的输入参数与跨步（Stride）。

3. **添加算子评测用例**

   根据实际需求，采用以下三种适配路径之一：

   - **路径 A：直接复用已有指标与输入生成器**
     若现有测试类的指标与输入生成规则完全满足需求，只需在算子列表字典或 `pytest.mark.parametrize` 参数列表中追加待测算子配置即可。

   - **路径 B：自定义输入生成函数（Custom Input Generator）**
     当指标统计通用但需要特定的数据分布或非常规参数时，可在类中传入自定义的 `input_fn`。例如 `test_special_perf.py` 中为 `topk` 算子定制的 `topk_input_fn`。

   - **路径 C：继承扩展专属 Benchmark 类**
     对于复杂算子（如 FlashAttention、MoE 分发对齐等），建议派生独立的 `Benchmark` 子类，定义专用的吞吐计算公式与复杂的参数生成逻辑。可参考 `benchmark/` 下各专用算子的子类实现。

{{% /steps %}}

## 使用模型 Profiling Shape 进行基准测试

FlagGems 在 `benchmark/models_shapes/` 下提供了按模型分类的 Profiling Shape 文件。
这些 YAML 文件记录了真实推理过程中采集到的张量 shape，以及每个算子在各推理阶段占总
GPU 时间的比例（`pct`）和调用次数（`calls`）。通过 `--model-profile` 参数可以直接
使用这些 shape 进行测试，并在输出表格中新增 **Time Share** 列，从而直观地确定优化优先级。

### 已收录的 Shape 文件

| 文件 | 模型 | 硬件 | 采集日期 |
|------|------|------|----------|
| `benchmark/models_shapes/285B.yaml` | 285B MoE（DeepSeek-V3.2 架构） | 8×H100 80GB，TP=8 | 2026-11-16 |

### 使用方式

```bash
# 使用 285B Profile 运行相关算子基准测试：
pytest benchmark/test_mm.py benchmark/test_bmm.py \
       benchmark/test_rms_norm.py benchmark/test_add.py \
       benchmark/test_mul.py benchmark/test_copy.py \
       benchmark/test_fill.py benchmark/test_zeros.py \
       benchmark/test_silu.py benchmark/test_silu_and_mul.py \
       benchmark/test_softmax.py \
       --model-profile benchmark/models_shapes/285B.yaml --level core -s

# 或使用辅助脚本自动从 YAML 推导所需的测试 marker：
python benchmark/benchmark_for_models.py \
       --shape-file benchmark/models_shapes/285B.yaml
```

`--model-profile` 标志有两个作用：

1. **覆盖 `--shape_file`**：将 shape 来源切换为指定 YAML 文件，每个基准测试只运行
   YAML 中定义的 shape。YAML 中未列出的算子会被静默跳过，不影响其原有的 shape 行为。
2. **附加 Profiling 元数据**：终端输出新增 `Time Share (prefill/decode)` 列，格式如下：

   ```
   pf 10.99%/35520x | dc 23.86%/84760x
   ```

   其中 `pf` 表示 prefill 阶段，`dc` 表示 decode 阶段；`%` 为占当前阶段总 GPU
   时间的比例，`x` 为该 profiling 窗口内的调用次数。

### YAML 文件格式

每个 YAML 条目包含三个字段：

```yaml
mm:
  profiling:
    prefill:
      pct: 10.99    # 占 prefill 阶段总 GPU 时间的百分比
      calls: 35520  # profiling 窗口内的 kernel 调用次数
    decode:
      pct: 23.86
      calls: 84760
  shapes:
    - [1, 8192, 3072, 1536]   # [b, M, N, K]，BlasBenchmark (mm/bmm) 格式
    - [1, 8192, 4096, 2048]
  shape_desc: "b, M, N, K"
```

条目的键名（如 `mm`）须与对应 `Benchmark` 子类传入的 `op_name` 完全一致，
flag 才能生效。各 Benchmark 子类对应的 shape 格式如下：

| 子类 | Shape 格式 |
|------|-----------|
| `BlasBenchmark`（`mm`、`bmm`） | `[b, M, N, K]` |
| `GenericBenchmark2DOnly`（`rms_norm`） | `[M, N]` |
| `BinaryPointwiseBenchmark` / `GenericBenchmark`（`add`、`mul`、`copy_`、`silu_and_mul`） | `[M, N]` |
| `UnaryPointwiseBenchmark`（`silu`、`zeros`） | `[M, N]` |
| `GenericBenchmark` 标量版本（`fill_scalar`） | `[N]` |
| `UnaryReductionBenchmark`（`softmax`） | `[M, N]` |

### 添加新模型的 Shape 文件

1. 在 `benchmark/models_shapes/<ModelName>.yaml` 下参照上述格式新建文件。
2. 通过真实推理运行采集 Profiling 数据（例如使用开启 `record_shapes=True` 的
   `torch.profiler`）。
3. 对于 FlagGems 中已注册的算子（见 `src/flag_gems/__init__.py`），为其添加实测
   shape 和 GPU 时间占比条目。
4. 在上方**已收录的 Shape 文件**表格中补充新文件的说明行。
