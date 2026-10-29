---
title: 海光（Hygon）
weight: 20
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

# 海光（Hygon）DCU 后端优化指南

海光 DCU 加速卡上的 Triton Kernel 通过 HIP 编译器路径执行。切块大小（Tiling Size）、Warp 并发数、软件流水线深度（Pipeline Depth）以及 Grid 程序块排布顺序共同决定了物理寄存器、片上共享内存（LDS，Local Data Share）及 L2 缓存的利用效率。

本文以典型海光深度计算单元 Hygon BW / `gfx936` 为例，介绍核心硬件约束与对应的调优策略。

## 1. 硬件资源与调优边界

以 Hygon BW / `gfx936` 架构为例，其核心硬件资源配置如下：

| 硬件资源类别 | 物理配置规格 | 性能调优建议 |
| :----------- | :----------- | :----------- |
| **计算单元（CU）** | 80 个 CU，每个 CU 包含 4 个 SIMD 单元 | 启动 Grid 时应提供足够的 Program 块数量，避免小矩阵计算仅激活少数 CU 导致算力闲置。 |
| **Wavefront 宽度** | 64 Lane（每个 Wavefront 包含 64 个执行线程） | 结合切块大小综合选取 `num_warps`。注意 AMD/Hygon 的 Wave 宽度（64）是 NVIDIA Warp（32）的 2 倍。 |
| **Wave 驻留上限** | 每个 SIMD 单元最多并发驻留 10 个 Wave | 每个 CU 的理论并发上限为 40 个 Wave；受限于寄存器用量、LDS 显存分配及 Workgroup 数量，实际驻留数通常更低。 |
| **矢量寄存器（VGPR）** | 每个 SIMD 单元配有 192 KB 寄存器文件 | 过大的分块尺寸与较长的中间变量生命周期会显著加剧寄存器溢出风险，迫使可驻留 Wave 数量骤降。 |
| **片上共享内存（LDS）** | 每个 CU 独享 64 KiB LDS 存储 | Workgroup 内部共享。流水线多重缓冲及转置排布转换（Layout Transformation）需严控 LDS 总用量。 |

> [!NOTE]
> 该设备还支持 **Packed FP32** 向量指令，能在编译器适配时单条指令并发吞吐两个 FP32 数据。上述资源配额以 `gfx936` 为基准，不同代际的海光 DCU 芯片规格请以对应产品手册为准。

## 2. DUMMA 专用矩阵乘加指令形状

《DTK 26.04.1 DUMMA 使用手册》及头文件 `du_mma.h`/`du_mma.hpp` 规范了底层 `du::dumma::du_mma_sync` 矩阵乘加硬件指令所支持的 Tile 形状。单个 Wavefront 协同完成 $D = A \times B + C$ 的输出切片计算。

下表中列出的 `16 × 16 × K` 为**单次 DUMMA 指令级执行切块**；Triton 中的高层分块 `BLOCK_M/N/K` 则通过在 K 维度循环拼接多次 DUMMA 指令来实现：

| A、B 输入类型 | 累加器类型 | 指令级分块 $M \times N \times K$ | 支持的目标架构 |
| :------------ | :--------- | :------------------------------- | :------------- |
| FP32 | FP32 | `16 × 16 × 4` | `gfx926 / 928 / 936 / 938` |
| FP32 / TF32 | FP32 | `16 × 16 × 8` | `gfx928 / 936 / 938` |
| FP16 / BF16 | FP32 | `16 × 16 × 16` | `gfx928 / 936 / 938` |
| 有符号 / 无符号 INT8 | INT32 | `16 × 16 × 32` | `gfx928 / 936 / 938` |
| 有符号 / 无符号 INT4 | INT32 | `16 × 16 × 64` | `gfx936 / 938` |
| FP64 | FP64 | `16 × 16 × 4` | `gfx926 / 936 / 938` |
| FP8（E4M3 或 E5M2） | FP32 | `16 × 16 × 32` | `gfx938` |

输入数据位宽越低，单次硬件指令在 K 维度上可吞吐的元素量越大。在调优 `tl.dot` 时，建议通过离线反汇编确认编译器是否正确匹配并生成了目标架构的 DUMMA 指令。

## 3. 软件流水线优化：重叠与资源的权衡

对于沿 K 轴累加循环的 GEMM 类 Kernel，通过调整 `num_stages` 可让编译器构建异步多阶段软件流水线，以异步预取数据掩盖全局内存读取的高延迟。

- **输入数据量快速估算**：以 FP16 精度矩阵乘为例，单轮加载 A、B 分块的数据体积约为 $(BLOCK\_M + BLOCK\_N) \times BLOCK\_K \times 2$ 字节。例如 `64 × 64 × 32` 对应 8 KiB 输入数据。
- **LDS 占用约束**：每个 CU 仅配有 64 KiB LDS。若单个 Workgroup 的流水线缓冲与对齐 Padding 消耗了 32 KiB LDS，则该 CU 最多只能并发驻留 2 个 Workgroup；一旦单组占用超过 32 KiB，CU 将只能串行容纳 1 个 Workgroup，整体并发隐蔽延迟能力显著受挫。

## 4. 矩阵分组调度（Grouped Launch）：提升 L2 缓存复用率

矩阵乘法通常将二维输出空间展平为一维 Grid。通过引入 `GROUP_M` 宏或分块逻辑，把连续的计算任务映射到同一行的连续输出块，可以让组内多个 M 分块共享复用 B 矩阵在 K 维度加载的数据，极大降低对全局显存 HBM 的重复带宽高频索取：

```python
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

> [!TIP]
> 缓存命中率与复用收益受矩阵长宽比与物理 CU 总数影响较大。在调优时建议将 `GROUP_M`（常用候选值如 4、8、16）与 `BLOCK_M/N/K` 共同纳入 LibTuner 搜索空间中综合测量。

## 5. 参考资料

- 《DCU 编程实战》（2026 版），第 6.8.4 节
- 《DTK 26.04.1 DUMMA 使用手册》，第 3 节表 3.1
- DTK 26.04 `du_mma.h`、`du_mma.hpp`（位于系统的 `/opt/dtk/include/`）
