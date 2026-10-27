---
title: 后端深度优化
weight: 45
bookCollapseSection: true
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

# 硬件后端深度优化指南

在异构硬件加速器上，Triton Kernel 的最优启动参数（Launch Configurations）与切块大小（Tiling Sizes）高度依赖物理硬件架构与编译器后端。在针对特定硬件进行性能攻关时，建议先充分理解目标芯片的存储层级与指令特征，再结合 [FlagGems 预调优流程](../usage/tuning.md) 在代表性业务 Shape 上进行精细搜索。

- **[华为昇腾（Ascend NPU）优化](ascend/)**：AI Core 的 Cube/Vector 计算与存储路径、片上存储（L0/L1/UB）对齐与预算规划、Triton-Ascend 编译参数及常见性能瓶颈调优。
- **[海光（Hygon DCU）优化](hygon/)**：DCU/HIP 存储层级、Wavefront 驻留上限、DUMMA 矩阵指令形状与分组调度优化。

> [!TIP]
> 文档中所推荐的分块配置与参数组合均为典型工况下的经验参考值。在正式合入或投产前，请务必在目标物理设备及当前编译器版本上验证数值一致性与实际端到端耗时。
