---
title: FlagGems 模块
weight: 80
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
# Building Custom Models Using Gems Operators
-->
# 基于 FlagGems 模块构建自定义模型

在部分前沿探索或模型研发场景中，算法工程师通常需要从零搭建新架构或对现有模型结构进行定制改动。为了支持这一需求，*FlagGems* 提炼并封装了一系列大语言模型（LLM）中核心的高性能神经网络模块（Modules）。

这些模块继承自标准的 `torch.nn.Module`，内部完全基于深度优化的 *FlagGems* 融合算子实现。开发者无需手写复杂的 CUDA/Triton 底层代码，只需像使用原生 PyTorch Layer 一样将其引入模型网络，即可直接享受极致的内核级性能加速。

所有模块源码均位于仓库的 [`src/flag_gems/modules`](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/modules) 目录下。

## 可用模块清单

| 模块类名 | 功能描述 | 支持特性 |
| :------- | :------- | :------- |
| `GemsRMSNorm` | 均方根层归一化（RMS LayerNorm） | 支持融合残差相加（Add-RMSNorm），支持 `inplace` 与 `outplace` 模式 |
| `GemsRope` | 经典旋转位置编码（RoPE） | 支持高吞吐向量化计算，支持 `inplace` 与 `outplace` 模式 |
| `GemsDeepseekYarnRoPE` | 基于 YaRN 插值外推的旋转位置编码（DeepSeek 风格） | 专为超长上下文外推优化，支持 `inplace` 与 `outplace` 模式 |
| `GemsSiluAndMul` | SwiGLU 结构中的 SiLU 激活与逐元素乘法融合 | 消除中间写回显存，仅支持 `outplace` 模式 |

开发者可直接将上述模块作为原生 PyTorch 层的直接替换（Drop-in Replacement）。更多诸如 Fused Attention、MoE 分发路由层与端到端 Transformer Block 正在持续扩充中。
