---
title: 功能特性概览
weight: 10
---

<!--
# Features Overview

## Rich Collection of Operators

FlagGems features a large collection of PyTorch compatible operators.
Refer to [operator supported](/FlagGems/references/operators/) and
[experimental operators](/FlagGems/references/experiental/)
for list of formally supported operators and experimental operators.
-->
# 功能特性概览

## 丰富的算子集合

*FlagGems* 的一大特性是提供丰富的 PyTorch 兼容算子集合。
参阅[支持的算子](/FlagGems/zh-cnreferences/operators/)和
[实验性算子](/FlagGems/zh-cn/references/experimental/)
页面了解正式支持的算子集合以及当前处于实验阶段的其他算子。

<!--
## Hand-optimized Performance for Selected Operators

The following chart shows the speedup of FlagGems compared with PyTorch ATen library
in eager mode. The speedup is calculated by averaging the speedup on each shape,
representing the overall performance of the operator.

![Operator Speedup](./assets/speedup-20251225.png)
-->
## 针对部分算子的选择性手工性能优化

下面的图表中显示的是 FlagGems 与 PyTorch ATen 算子库在 Eager 模式下的性能比较。
所展示的数据是针对不同数据形状下加速比的平均值，代表了算子的总体性能。

![算子加速比](/FlagGems/images/speedup-20251225.png)

<!--
## Eager-mode ready, independent of `torch.compile`

> TBD
-->
## 直接进入 Eager 模式，无需 `torch.compile`

> TODO: 内容待补充。

<!--
## Automatic Code Generation

FlagGems provides an automatic code generation mechanism that enables developers
to easily generate both pointwise and fused operators.
The auto-generation system supports a variety of requirements, including standard
element-wise computations, non-tensor parameters, and specifying output data types.
For more details, please refer to the [pointwise dynamic](/FlagGems/contribution/pointwise_dynamic/)
documentation.
-->
## 自动代码生成

*FlagGems* 提供一种自动代码生成的机制，方便开发者生成逐点（pointwise）
算子和融合（fused）算子。代码自动生成系统能够满足多种不同的需求，
包括标准的逐元素（element-wise）计算、非张量参数以及设定输出数据类型等等。
参阅[逐点动态算子](/FlagGems/zh-cn/overview/pointwise-dynamic/)文档以了解更多细节。

<!--
## Function-level kernel dispatching

FlagGems introduces `LibEntry`, which independently manages the kernel cache and
bypasses the runtime of `Autotuner`, `Heuristics`, and `JitFunction`.
To use this feature, simply decorate the Triton kernel with `LibEntry`.
-->
## 函数层级内核派发

*FlagGems* 提供 `LibEntry` 来独立管理对内核的缓存，从而越过 `Autotuner`、
`Heuristics` 和 `JitFunction` 等运行时处理机制。要使用这一特性，
你可以为对应的 Triton 内核添加 `libentry` 修饰符。

<!--
`LibEntry` also supports direct wrapping of `Autotuner`, `Heuristics`, and `JitFunction`,
preserving full tuning functionality. However, it avoids nested runtime type invocations,
eliminating redundant parameter processing. This means no kneed for binding or type wrapping,
resulting in a simplified cache key format and reduced unnecessary key computation.
-->
`LibEntry` 类也支持对 `Autotuner`、`Heuristics` 和 `JitFunction` 的直接封装，
从而完整地保留性能调优能力。不过，`LibEntry` 能够避免嵌套的运行时类型调用，
消除冗余的参数处理操作。这就意味着，开发者不需要绑定或者封装类型信息，
所使用的将是一种简化的缓存键格式，减少了不必要的键值计算操作。

<!--
### Multi-backend hardware support

FlagGems supports a wide range of hardware platforms (10+ backends) and
has been extensively tested across different hardware configurations.
For a comprehensive and up to date list of supported platforms,
please check [platforms supported](/FlagGems/overview/platforms/).
-->
### 多种后端硬件支持

*FlagGems* 支持很多种硬件平台（后端超过 10 种），并且已经在不同硬件配置环境下通过了大量测测试。
参阅[平台支持](/FlagGems/zh-cn/overview/platforms/)了解完整的、最新的支持平台列表。

<!--
### C++-wrapped operators

*FlagGems* can be installed either as a pure Python package or as a package with C++ extensions.
The C++ runtime is designed to address the overhead of the Python runtime
and improve end-to-end performance.

There is an ongoing work that implements the Triton function dispatcher
in C++ language. Stay tuned.
-->
### C++ 封装的算子

*FlagGems* 可以作为纯 Python 包来安装，也可以附带 C++ 扩展支持特性来安装。
其中的 C++ 运行时被设计用来解决 Python 运行时的性能开销问题，
旨在提高最终系统的端到端性能。

开发团队目前正在尝试用 C++ 语言实现 Triton 的函数派发程序。敬请留意。
