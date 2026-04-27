---
title: 项目源码结构
weight: 40
---

<!--
# Project Structure
-->
# 项目源码结构

```none
FlagGems
├── src                  // python 源代码
│   └──flag_gems
│       ├──utils         // python 自动代码生成工具
│       ├──ops           // python 单个算子
│       ├──fused         // python 融合算子
│       `──testing       // python 测试工具
├── tests                // python 单元测试用例
├── benchmark            // python 性能测试用例
├── examples             // python 模型测试文件
├── cmake                // 用于 C++ 扩展的 c++ CMake 文件
├── include              // C++ 头文件
├── lib                  // 用于实现算子的C++ 源代码
├── ctest                // C++ 测试用例文件
├── triton_src           // Triton JIT 函数临时源码目录
├── docs                 // 文档
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── ...
```
