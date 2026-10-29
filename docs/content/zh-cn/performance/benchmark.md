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
