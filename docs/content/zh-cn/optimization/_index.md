---
title: 后端优化
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

# 后端优化

Triton 启动参数和合适的分块大小取决于设备与编译器。先了解目标后端的硬件结构和编译参数，再按[FlagGems 调优流程](/FlagGems/zh-cn/usage/tuning/)测量实际业务形状。

- [Ascend](ascend/)：AI Core 的计算与存储路径、Triton-Ascend 参数和 NPU 调优。
- [Hygon](hygon/)：DCU/HIP 存储层次、Triton 启动参数和分块调优。

文中的配置是候选值；应用到目标设备和已安装的编译器前，应验证数值正确性与耗时。
