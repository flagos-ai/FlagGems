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

昇腾的 Cube 路径执行矩阵乘，Vector 路径执行逐元素计算和归约；Scalar 路径负责地址计算、循环控制等。数据从全局内存（GM，通常为 HBM）经 MTE 搬运到片上缓冲区。Cube 使用 L1、L0A、L0B 和 L0C；Vector 使用统一缓冲区（UB）。**UB 与 L0 是不同的存储空间，L1 是 Cube 侧片上缓冲区，不是跨核共享的 L2 缓存。** 包含 `tl.dot` 和 Vector 后处理的 kernel 可能需要在两条路径间搬运中间结果并同步。

下表为 Triton-Ascend/AscendNPU-IR 资料中列出的片上容量，单位为 KiB。GM/HBM 容量、带宽和 L2 大小随具体设备变化，不能由这张表推导；这些层级的延迟也没有跨型号通用的固定值。

| 存储空间 | IR 标识 | 910B / 910_93 | 910_95 / 950 系列 | 主要用途 |
| --- | --- | ---: | ---: | --- |
| GM | `gm` | 设备决定 | 设备决定 | 全局数据；可能经过 L2 缓存 |
| L1 | `cbuf` | 512 KiB | 512 KiB | Cube 输入缓存；通常要求 32 B 对齐 |
| L0A / L0B | `ca` / `cb` | 各 64 KiB | 各 64 KiB | 矩阵 A/B 输入缓存 |
| L0C | `cc` | 128 KiB | 256 KiB | 矩阵乘累加结果；相关搬运按 512 B 对齐规划 |
| UB | `ub` | 192 KiB | 248 KiB 可用（标称 256 KiB，预留 8 KiB） | Vector 工作区；通常要求 32 B 对齐 |
| BT / FP Buffer | — | 1 / 7 KiB | 1 / 7 KiB | Bias 表 / FixPipe 中间缓冲 |

典型 Cube 数据流为 `GM → L1 → L0A/L0B → L0C → GM`，Vector 数据流为 `GM → UB → GM`。910_95/950 系列还支持通过 FixPipe 将 L0C 结果直接送入 UB，以及将 UB 数据送回 L1；910B/910_93 的相应混合路径通常需要经过 GM。同步方式和可用通路须以目标芯片与编译器为准。

### 核心数与 Grid

核心数按型号和计算路径分别查询。参考规格中，910B1/B2 为 **24 Cube / 48 Vector**，910B3/B4 为 **20 Cube / 40 Vector**；910_95 系列还随具体 SKU 变化。不要把 Cube 与 Vector 数量相加作为单个 kernel 的 grid。FlagGems 通过 Triton 设备属性中的 `num_vectorcore` 获取向量核数；矩阵 kernel 可查询 `num_aicore`。

```python
import torch
import torch_npu  # 注册 torch.npu
import triton.runtime.driver as driver

device_index = torch.npu.current_device()
props = driver.active.utils.get_device_properties(device_index)
vector_cores = props["num_vectorcore"]
cube_cores = props["num_aicore"]
```

普通 Triton-Ascend 启动采用 `(x,)`、`(x, y)` 或 `(x, y, z)` tuple，最多三维；默认路径的 `coreDim = x * y * z` 不应超过 **65535**。超过时会出现 `coreDim ... can't be greater than UINT16_MAX`。1D grid 通常便于按目标路径的核数分配工作；2D/3D 的乘积同样计入 `coreDim`，但“乘积必须小于物理核数”是调优选择，并非 Grid 语法限制。

大 shape 可将 grid 限制为 `min(triton.cdiv(N, BLOCK_SIZE), vector_cores)`，在每个 program 内按 `tl.num_programs(0)` 跨步处理多个块：

```python
import triton
import triton.language as tl

@triton.jit
def kernel(x, y, N, BLOCK_SIZE: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(N, BLOCK_SIZE), tl.num_programs(0)):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + offsets, mask=offsets < N, other=0)
        tl.store(y + offsets, values, mask=offsets < N)

grid = (min(triton.cdiv(N, BLOCK_SIZE), vector_cores),)
kernel[grid](x, y, N, BLOCK_SIZE)
```

`TRITON_ALL_BLOCKS_PARALLEL=1` 可在部分工具链上自动收缩逻辑 grid；仅在各 program 无执行顺序依赖、原子操作或跨核同步时考虑使用，并先验证正确性。

### 分块容量与对齐

910B 的 192 KiB UB 若只存放**一个** tile，按字节数计算的理论上限为 fp16/bf16 98,304 个元素或 fp32 49,152 个元素。这不是 Triton `BLOCK_SIZE` 的许可上限，也不是 matmul 的实际可用 tile 大小：同时存活的输入、输出、中间值、对齐填充和多缓冲都会占空间；编译器还可能有更小的单块限制。910_95 的 248 KiB 可用 UB 也需按同样方式核算。遇到 `ub overflow`（报错常以 bit 为单位）时，减小 `BLOCK_*`，或用核内 `BLOCK_SIZE_SUB` 循环切分。

Cube 路径应检查尾轴和矩阵行宽对齐。512 B 行宽对应 fp16/bf16 256 个元素或 fp32 128 个元素；不足时可能补齐并增加搬运，不宜把所有不满足条件的分块都判为硬件错误。可从 `BLOCK_M/N=128/256`、`BLOCK_K=128/256` 等候选开始搜索，但必须计入 A/B/C 的独立缓冲和目标芯片支持的数据类型。参考资料将 A2/A3 的 FP8 `tl.dot` 列为不支持，而 910_95 仅在特定 `tl.dot_scaled` 路径支持 FP8；不能仅按 1 字节元素大小推定可用 tile。

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

910_95/950 系列的参考编译器默认关闭 `multibuffer`，需要实验时应显式传入 `multibuffer=True`。双缓冲会增加同时存活的片上数据；若只有两个等大的 UB 缓冲且忽略其他开销，单份可用空间约减半，但实际容量须看编译器分配报告。

### 常见性能问题与处理

| 现象 | 检查与处理 |
| --- | --- |
| `aiv_scalar_ratio` 偏高 | 检查生成 IR 中的标量循环。910B/910_95 上 i64 向量算术、部分整数比较可能降级；值在 int32 范围内时可改用 int32。对需改用 fp32 的比较，先确认整数值在 fp32 精确表示范围内；int32 的相等比较不应一律转换。 |
| mask 加载阻碍存算重叠 | 只有 padding 值**完全不会参与**后续计算、归约或无 mask 的 store 时，才测试 `tl.load(..., care_padding=False)`；否则保留正确填充值。 |
| 非连续访存 | 调整索引或分块以合并访问；`.contiguous()` 会额外复制，应计入端到端耗时再决定是否预处理。 |
| 大矩阵出现 L2 冲突或复用差 | 比较常规、分组和对角映射，按 profiling 选用；不要将对角 tiling 设为所有矩阵的固定规则。参考实现提供 `grouped_launch_diagonal`；避免在 Ascend 路径直接使用不受支持的 `tl.swizzle2d`。 |
| 低精度累加误差或溢出 | fp16/bf16 输入的归约和矩阵累加优先检查 fp32 accumulator，并核对最终容差。 |

### 显式缓冲与同步扩展

需要自行管理 Cube/Vector 数据通路时，安装版本的 `triton.backends.ascend.language.cann.extension.core` 可提供 `ascend_address_space`（GM/L1/UB/L0A/L0B/L0C）、`copy`、`fixpipe`、`sync_block_set/wait/all` 和 `sub_vec_id/num`。`fixpipe` 的 L0C→UB 直通仅适用于支持该通路的 910_95/950；同步事件须成对使用，事件编号按该版本约束选择。先确认扩展 API、设备能力和生成的搬运路径，再用于实际 kernel。

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
- [cannbot-knowledge：Grid 与 program ID](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/grid_and_program_id.md)
- [cannbot-knowledge：硬件规格 IR 映射](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/apis/ir/reference/hardware_specs_ir.md)
- [cannbot-knowledge：内存层次](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/apis/ir/architecture/memory_hierarchy.md)
- [cannbot-knowledge：标量降级与 padding](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/scalar_degradation_avoidance.md)
- [cannbot-knowledge：`care_padding` 适用条件](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/care_padding.md)
- [cannbot-knowledge：NPU 数据类型与对齐](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/data_types_ascend.md)
- [cannbot-skills：Ascend 编译选项](https://gitcode.com/cann/cannbot-skills/blob/affd5a88dd956022a598adf2f5ed9f3e26718d48/ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/04-Compilation-Pipeline/07-compile-options.md)
- [cannbot-skills：Ascend 内存模型与扩展 API](https://gitcode.com/cann/cannbot-skills/blob/affd5a88dd956022a598adf2f5ed9f3e26718d48/ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/01-Programming-Model/03-memory-model.md)
- [Triton-Ascend 后端编译器（`NPUOptions`）](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/backend/compiler.py)
