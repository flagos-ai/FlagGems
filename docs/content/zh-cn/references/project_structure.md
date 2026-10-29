---
title: 项目源码结构
weight: 40
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
# Project Structure
-->
# 项目源码结构

```none
FlagGems
├── src/flag_gems         // Python 核心算子库源码
│   ├── utils             // 动态代码生成、调优与公共工具
│   ├── ops               // 基础逐点与常规算子实现
│   ├── fused             // 融合算子实现（FlashAttention、MoE 等）
│   ├── modules           // PyTorch nn.Module 级神经网络高阶组件
│   ├── runtime           // 跨芯片后端适配与分发运行时
│   └── testing           // 内部精度与测试辅助工具
├── tests                 // 算子单元精度测试用例（pytest）
├── benchmark             // 算子性能基准评测用例与自动化脚本
├── examples              // 典型大模型前向推理与预调优示例
├── cpp                   // C++ 扩展源码及构建脚本
│   ├── include           // C++ 对外头文件
│   ├── lib               // C++ 算子实现源码
│   └── ctest             // C++ 原生单元测试用例
├── conf                  // 算子目录清单与多后端配置（operators.yaml 等）
├── docs                  // 项目文档站源码（Hugo）
├── tools                 // CI 校验、构建辅助及安装脚本
├── LICENSE               // Apache 2.0 开源许可协议
├── README.md             // 项目主介绍
└── CONTRIBUTING.md       // 社区贡献指南
```
