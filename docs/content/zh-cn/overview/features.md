---
title: 功能特性概览
weight: 10
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
# Features Overview

## Rich Collection of Operators

FlagGems features a large collection of PyTorch compatible operators.
Refer to [operator supported](../references/operators.md) and
[experimental operators](../references/experimental.md)
for list of formally supported operators and experimental operators.
-->
# 功能特性概览

## 丰富的算子生态

*FlagGems* 提供了涵盖大模型训练与推理主流场景的丰富 PyTorch 兼容算子库。
详情请参阅[正式支持算子清单](../references/operators.md)与[实验性算子清单](../references/experimental.md)。

## 关键算子的手工极致优化

下图展示了在 Eager 模式下，*FlagGems* 算子相比 PyTorch ATen 原生算子实现的加速比。数据通过对不同典型 Tensor Shape 的加速比取平均计算得出，直观反映了算子的端到端综合性能优势：

![算子加速比](../../../assets/speedup-20251225.png)

## 原生 Eager 模式支持，脱离 `torch.compile` 依赖

*FlagGems* 算子能够直接注册到 ATen 并在原生的 Eager 模式下无缝运行，无需提前经过 `torch.compile` 的编译追踪流程，极大简化了调试与部署流程，即插即用。

## 算子自动代码生成

*FlagGems* 提供了强大的算子代码自动生成机制，帮助开发者低门槛生成高质量的逐点（Pointwise）算子和融合（Fused）算子。自动生成系统全面支持标准逐元素计算、混合标量/非张量参数、动态输出类型推断等多样化需求。
更多详情请参阅[逐点动态算子](pointwise-dynamic.md)文档。

## 函数级 Kernel 分发调度

*FlagGems* 引入了 `LibEntry` 机制，独立管理 Kernel 缓存并绕过 Triton 默认 `Autotuner`、`Heuristics` 与 `JitFunction` 的部分运行时开销。开发者只需使用 `@libentry` 修饰对应的 Triton Kernel 即可享受到此特性。

`LibEntry` 还支持直接包装 `Autotuner`、`Heuristics` 和 `JitFunction`，在完整保留参数调优能力的同时，消除嵌套运行时的层层类型转换与冗余参数校验，从而大幅简化缓存键（Cache Key）的生成逻辑并降低无谓的计算开销。

## 广泛的多硬件平台支持

*FlagGems* 已支持超过 10 种主流 AI 加速芯片及硬件后端，并在多种硬件架构及异构集群中完成了深入的精度与性能验证。
有关各厂商芯片的适配程度与数据类型支持情况，请参阅[硬件平台支持](platforms.md)。

## C++ 扩展与轻量级分发

*FlagGems* 既可作为纯 Python 包开箱即用，也支持构建 C++ 高性能扩展。C++ 运行时专为消除 Python 解释器的 Dispatch 开销而设计，在端到端高频小算子调用场景下能带来显著的延迟降低与吞吐提升。目前团队也在持续推进基于 C++ 的底层 Triton 函数分发器，敬请期待。
