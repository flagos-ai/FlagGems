---
title: Ascend
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

# Ascend 优化

## 硬件架构

昇腾 AI Core 提供不同的 **Cube** 矩阵计算路径和 **Vector** 向量计算路径。矩阵分块使用 Cube 及其 L1/L0 缓冲区，逐元素计算和归约主要使用 Vector 及其统一缓冲区（UB）。全局内存搬运和同步连接这些路径。核心数量和片上存储容量因芯片代际而异，设置 grid 或分块容量前应查询目标设备。包含 `tl.dot` 和 Vector 后处理的 kernel 可能同时使用两条路径，并需要额外的中间存储。

## Triton 编译与启动参数

FlagGems 将 Ascend 配置为 NPU 后端（`device_name="npu"`），并使用已安装的 Triton/FlagTree 扩展。可通过 `kernel[grid](...)` 的关键字参数或 `triton.Config` 传入受支持的选项。扩展参数的可用性和默认值随 Triton-Ascend 版本变化；使用前应检查对应版本的 `NPUOptions` 定义。

| 参数 | 调整对象 | 适用场景 |
| --- | --- | --- |
| `BLOCK_SIZE`、`BLOCK_M/N/K` | 编译期分块大小（`tl.constexpr`），不是后端选项 | 平衡分核、UB/L1 占用和尾块 mask。 |
| `num_warps` | 编译器并行与布局选择 | 结合分块实测；SIMD 路径中的含义不同于 GPU warp 调度。 |
| `num_stages` | 编译器流水线设置 | 以安装版本的行为为准；FlagGems 的 Ascend 配置包含不同 stage 值。 |
| `compile_mode` | 选择编译路径（可用时为 `simd`、`unstructured_in_simt` 或 `simt_only`） | 先用安装版本的默认值；仅对合适的 workload 测试 SIMT，并验证输出。 |
| `multibuffer` | 搬运与计算重叠 | 有循环的 kernel 可以测试，并检查本地缓冲占用；默认值可能因架构而异。 |
| `enable_flatten` | 展平适用的循环结构 | 可在 Vector kernel 上尝试，再检查生成代码和耗时。 |
| `enable_mixed_cv`、`sync_solver`、`enable_auto_bind_sub_block` | Cube/Vector 协作与同步 | 仅在确实混合两条路径时考虑组合使用，先验证正确性。 |
| `enable_fp_fusion` | 浮点融合 | 更改后重新验证数值误差。 |

例如，逐元素 kernel 可将 `kernel[grid](..., BLOCK_SIZE=1024, num_warps=4)` 与相邻分块比较。有循环的 kernel 可将 `multibuffer=True` 与基线比较。无需给所有 kernel 默认启用 Cube/Vector 参数。

## 优化步骤

1. 判断 kernel 属于 Vector、Cube 还是混合路径；用代表性形状建立正确性与耗时基线。
2. 尽量让相邻 `tl.load`/`tl.store` 地址连续。尾块使用 mask，并按实际数据类型和布局检查对齐。
3. 扫描 grid 和分块。将同时存活的输入、中间值和额外缓冲计入片上存储预算；编译提示 UB 压力时先缩小分块。
4. 对循环 workload 测试 `multibuffer`；对 Cube/Vector 混合 workload 先检查同步和中间数据搬运，再尝试混合路径参数。
5. 利用设备 profiling 区分访存、Vector/Cube 利用率和启动开销。改变精度或融合设置后重新验证数值结果。

## 参考资料

- [cannbot-knowledge：NPU 编译参数](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/compile_params.md)
- [cannbot-knowledge：性能优化总览](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/perf_optimization_overview.md)
- [cannbot-knowledge：Tiling 策略](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/tiling.md)
- [cannbot-skills：Ascend 编译选项](https://gitcode.com/cann/cannbot-skills/blob/affd5a88dd956022a598adf2f5ed9f3e26718d48/ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/04-Compilation-Pipeline/07-compile-options.md)
- [Triton-Ascend 后端编译器（`NPUOptions`）](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/backend/compiler.py)
