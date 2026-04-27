---
title: 概览
weight: 10
---

<!--
# Performance Benchmarking Overview
-->
# 性能测试概览

<!--
*FlagGems* operators in general provides better or at least comparable performance
when compared to operators from the native PyTorch library.
We use the `triton.testing.do_bench` from the Triton project for benchmarking.
The kernel data obtained are shown in the following graph.
-->
与原生的 PyTorch 库中的算子相比，*FlagGems* 算子一般而言能够提供更好的、
至少是可比较的性能。
我们使用来自 Triton 项目的 `triton.testing.do_bench` 框架来执行性能基准测试。
下图展示的即是所获得的内核性能数据。

![算子加速比](/FlagGems/images/speedup-20251225.png)

<!--
The chart above shows the speedup of *FlagGems* compared with the PyTorch ATen library
in eager mode. The speedup is calculated by averaging the speedup on each shape,
representing the overall performance of the operator.
-->
上图中展示的是 *FlagGems* 与 PyTorch ATen 库在 Eager 模式下获得的性能加速比。
加速比数据是基于不同数据形状所获得的值的均值，代表了算子的整体性能。

<!--
To ensure that the performance of any new operators are within an acceptable range,
we require all contributions to the operators' inventory provide performance data.
You can benchmark your new operators (and the existing ones) using the *benchmark*
framework in *FlagGems*.

Check [operator benchmark](/FlagGems/performance/benchmark/) for instructions
on benchmark testing your operators.
-->
为了确保所有新增的算子的性能处于可接受范围，我们要求对算子库的所有贡献都要提供性能数据。
你可以使用 *FlagGems* 中的 *benchmark* 框架来对自己新编写的（以及现有的算子）
执行性能基准测试。

关于如何对算子执行性能基准测试的详细指令，
请参阅[算子性能基准测试](/FlagGems/zh-cn/performance/benchmark/)文档。
