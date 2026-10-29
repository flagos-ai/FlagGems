---
title: 概述
weight: 5
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
# Overview
-->
# 使用指南概览

*FlagGems* 提供了丰富而灵活的使用模式。不同模式之间互不排斥，您可以根据具体的业务需求、调优目标或部署上下文自由组合使用：

- **[基础用法与启用](basic.md)**

  在运行模型之前启用 `flag_gems`。支持全局一键替换，也支持使用上下文管理器在特定代码块内局部生效。

- **[选择性启用算子](selective.md)**

  若需针对特定算子进行精细化控制（如仅加速高收益算子，或将个别算子回退到原生实现），可通过包含/排除列表进行按需注册。

- **[算子调试与日志追踪](debugging.md)**

  在需要排查算子分发、定位执行路径或获取详细执行耗时时，可配置调试选项输出完整的算子调用链跟踪日志。

- **[体验实验性算子](experimental.md)**

  *FlagGems* 算子库持续孕育前沿的高性能算子实现。对于尚在实验阶段的算子，也可按需显式启用尝鲜。

- **[在非 NVIDIA 异构平台上使用](non-nvidia.md)**

  *FlagGems* 致力于消除不同 AI 芯片之间的移植壁垒。您可以了解针对昇腾、摩尔线程、沐曦、天数智芯等国产/异构硬件的适配指南。

- **[在多 GPU 及分布式环境中运行](distributed.md)**

  在多卡并行训练（如 Megatron-LM）或分布式大模型推理框架（如 vLLM）中，指导您正确完成进程与设备绑定，平稳启用 *FlagGems*。

- **[与主流框架无缝集成](frameworks.md)**

  介绍如何与 [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)、[vLLM](https://docs.vllm.ai/en/latest/)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 等业界主流框架协同工作。

- **[基于 FlagGems 构建自定义模型模块](modules.md)**

  *FlagGems* 提供了诸如 RMSNorm、RoPE、FlashAttention、MoE 等高阶深度学习模型组件，便于直接集成到新模型或现有模型骨架中。

- **[启用预调优（Pre-tuning）获取极致性能](tuning.md)**

  *FlagGems* 提供了针对 Triton 调优流程轻量级增强的 [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/utils/libentry.py#L206)，有效消除在线运行时调优开销。

- **[使用 C++ 封装算子获得极致吞吐](cpp.md)**

  针对高频调度开销敏感的关键算子，*FlagGems* 提供了底层 C++ 原生扩展分发实现，进一步榨干硬件性能。
