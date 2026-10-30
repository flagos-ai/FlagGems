---
title: 华为昇腾（Ascend）
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

# 华为昇腾（Ascend）后端优化指南

## 1. 硬件架构与存储层级

昇腾 AI Core 采用多计算单元异构设计：**Cube 路径**负责矩阵乘法，**Vector 路径**执行逐元素（Pointwise）计算与归约操作，**Scalar 路径**负责地址计算与循环控制。
数据从全局内存（GM，通常为 HBM）经内存传输引擎（MTE）搬运到片上缓冲区。Cube 单元专属使用 L1、L0A、L0B 和 L0C；Vector 单元使用统一缓冲区（UB，Unified Buffer）。

> [!IMPORTANT]
> **UB 与 L0 是完全独立的物理存储空间**；**L1 是 Cube 侧专用片上缓冲区**，并非跨核共享的 L2 缓存。包含 `tl.dot` 及 Vector 后处理的复合 Kernel 通常需要在两条路径之间搬运中间数据并进行跨通路同步。

下表列出 Triton-Ascend / AscendNPU-IR 规范中各典型型号的片上存储容量（单位：KiB）。全局内存 GM/HBM 容量、带宽及 L2 缓存大小由具体芯片型号决定：

| 存储空间 | IR 标识 | 910B / 910_93 | 910_95 / 950 系列 | 核心用途与对齐要求 |
| :------- | :------ | :------------ | :---------------- | :----------------- |
| GM | `gm` | 设备具体规格 | 设备具体规格 | 全局数据；访问经由 L2 缓存 |
| L1 | `cbuf` | 512 KiB | 512 KiB | Cube 输入缓存；通常要求 32 B 对齐 |
| L0A / L0B | `ca` / `cb` | 各 64 KiB | 各 64 KiB | 矩阵 A/B 操作数片上输入缓存 |
| L0C | `cc` | 128 KiB | 256 KiB | 矩阵乘累加结果；搬运按 512 B 对齐规划 |
| UB | `ub` | 192 KiB | 248 KiB 可用（标称 256 KiB，系统预留 8 KiB） | Vector 工作区；通常要求 32 B 对齐 |
| BT / FP Buffer | — | 1 / 7 KiB | 1 / 7 KiB | Bias 表 / FixPipe 中间缓冲 |

- **典型 Cube 数据流**：`GM → L1 → L0A/L0B → L0C → GM`
- **典型 Vector 数据流**：`GM → UB → GM`

910_95/950 系列进一步支持通过 FixPipe 将 L0C 累加结果直接搬运进 UB，以及将 UB 数据写回 L1；而 910B/910_93 芯片上的混合路径通常需要途经全局内存（GM）。

### 1.1 核心数与 Grid 映射

昇腾芯片的物理核心数按计算路径分别呈现。以主流参考规格为例：
- **910B1 / 910B2**：24 Cube / 48 Vector
- **910B3 / 910B4**：20 Cube / 40 Vector
- **910_95 系列**：根据具体产品 SKU 动态配置

> [!WARNING]
> 切勿将 Cube 与 Vector 核心数量相加作为单个 Kernel 的 Grid 尺寸。

*FlagGems* 通过 Triton 设备属性中的 `num_vectorcore` 获取向量核数；矩阵类 Kernel 则可查询 `num_aicore`：

```python
import torch
import torch_npu  # 注册 npu 设备后端
import triton.runtime.driver as driver

device_index = torch.npu.current_device()
props = driver.active.utils.get_device_properties(device_index)
vector_cores = props["num_vectorcore"]
cube_cores = props["num_aicore"]
```

在 Triton-Ascend 中，启动 Grid 采用 1D/2D/3D 元组，默认路径下 `coreDim = x * y * z` 不能超过 **65535**（否则将报 `coreDim ... can't be greater than UINT16_MAX` 异常）。

处理大规模 Tensor Shape 时，建议将 Grid 限制为物理核心数，并在每个 Program 内部按 `tl.num_programs(0)` 跨步循环（Grid-Stride Loop）处理多个数据块：

```python
import triton
import triton.language as tl


@triton.jit
def kernel(x, y, N, BLOCK_SIZE: tl.constexpr):
    for block in range(
        tl.program_id(0), tl.cdiv(N, BLOCK_SIZE), tl.num_programs(0)
    ):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + offsets, mask=offsets < N, other=0)
        tl.store(y + offsets, values, mask=offsets < N)


grid = (min(triton.cdiv(N, BLOCK_SIZE), vector_cores),)
kernel[grid](x, y, N, BLOCK_SIZE)
```

### 1.2 分块容量与内存对齐

- **UB 预算**：910B 的 192 KiB UB 若存放单个分块，理论上限约为 FP16/BF16 98,304 个元素或 FP32 49,152 个元素。但实际计算中，输入、输出、中间变量、内存对齐填充与多缓冲（Multibuffer）会同时占用空间。若遇到 `ub overflow` 错误（通常以 bit 为单位报警），应主动缩减 `BLOCK_*` 或在核内使用 `BLOCK_SIZE_SUB` 循环切分。
- **Cube 对齐**：Cube 路径需关注尾轴和矩阵行宽对齐。512 B 行宽对应 FP16/BF16 256 个元素或 FP32 128 个元素；不足时硬件或编译器可能引入自动 Padding 并增加无效搬运。

## 2. Triton 编译与启动参数

*FlagGems* 将 Ascend 配置为 NPU 后端（`device_name="npu"`），使用系统安装的 Triton / FlagTree 扩展。可通过 `kernel[grid](...)` 关键字参数或 `triton.Config` 传入对应调优选项：

| 参数名称 | 调整对象 | 推荐场景与调优建议 |
| :------- | :------- | :----------------- |
| `BLOCK_SIZE`、`BLOCK_M/N/K` | 编译期分块大小（`tl.constexpr`） | 平衡核心并发度、UB/L1 片上显存占用以及尾块 Mask 开销 |
| `num_warps` | 编译器并行度与布局选择 | 结合分块实测。SIMD 路径中的语义不同于 GPU Warp 调度 |
| `num_stages` | 编译器流水线深度 | 调节数据预取重叠。以当前安装的 Triton-Ascend 行为为准 |
| `compile_mode` | 编译模式选择（`simd` / `simt_only` 等） | 默认优先使用 `simd`；对适宜的计算可尝试 SIMT 并校验数值精度 |
| `multibuffer` | 搬运与计算双缓冲重叠 | 存在循环搬运的 Kernel 可开启测试；需警惕 UB 占用翻倍风险 |
| `enable_flatten` | 循环展平展开优化 | 适用于简单的 Vector Kernel，可有效消除循环分支跳转开销 |
| `enable_mixed_cv`、`sync_solver` | Cube 与 Vector 协作同步 | 仅在融合了矩阵乘与向量计算的复合 Kernel 中按需启用 |
| `enable_fp_fusion` | 浮点计算融合（FMA 等） | 开启后需严格比对浮点数值容差 |

> [!NOTE]
> 在 910_95/950 系列上，默认编译策略通常关闭 `multibuffer`。若需测试数据搬运与计算重叠，可显式指定 `multibuffer=True`。注意双缓冲将使活跃数据空间占用接近翻倍。

## 3. 常见性能瓶颈排查

| 典型性能瓶颈 | 排查定位与应对策略 |
| :----------- | :----------------- |
| **`aiv_scalar_ratio` 标量占比过高** | 检查生成的底层 IR 是否存在标量循环。在 910B 上，`int64` 向量计算及部分整数比较可能发生降级；若数值范围满足，应优先转为 `int32`。 |
| **Mask 加载阻碍存算重叠** | 仅当 Padding 区域的垃圾数据**完全不会**污染后续归约累加或无 Mask 的写回时，才考虑使用 `tl.load(..., care_padding=False)`；否则切勿冒险。 |
| **非连续与跨步访存** | 尽量调整多维索引计算或切块以实现连续合并访存。谨慎使用 `.contiguous()`，因为显式全局内存拷贝往往会抵消算子带来的收益。 |
| **大矩阵 L2 Cache 冲突或复用率低** | 尝试使用行优先分组映射（Grouped Launch）；但注意避免在 Ascend 上直接调用未经支持的 `tl.swizzle2d`。 |
| **低精度累加溢出** | 处理 FP16/BF16 输入的归约和矩阵乘法时，中间累加器（Accumulator）必须使用 FP32 精度，避免发散。 |

## 4. 推荐优化流程

1. **识别路径类型**：确认算子属于 Vector 纯向量、Cube 纯矩阵还是混合路径；选取典型业务 Shape 建立基准耗时与精度对照。
2. **保障内存连续性**：优化内存访问步长，确保相邻线程/指令的访存地址连续并满足对齐约束。
3. **精细规划切块（Tiling）**：根据 UB（192 KiB / 248 KiB）和 L1（512 KiB）的物理预算合理确定分块，避免触碰溢出阈值。
4. **探索重叠与流水线**：为带循环的 Kernel 探索 `multibuffer=True` 与 `num_stages` 参数。
5. **Profiling 分析验证**：结合 Ascend Profiler（昇腾分析器）分析算力利用率（Cube/Vector MTE 占比），确保在性能提升的同时通过数值精度门禁。

## 5. 参考资料

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
- [Triton-Ascend：环境变量与编译选项参考](https://triton-ascend.readthedocs.io/zh-cn/v3.6.0/environment_variable_and_compiler_options_reference.html#id1)
