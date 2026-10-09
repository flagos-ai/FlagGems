---
title: Hygon
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

# Hygon 优化

海光 DCU 的 Triton kernel 通过 HIP 路径执行。分块大小、warp 数、流水线深度和程序块顺序决定寄存器、片上共享内存（LDS）及缓存的使用方式。下文以 Hygon BW / `gfx936` 为例，介绍与这些资源直接相关的优化机制。

## 硬件与配置边界

以本文使用的 Hygon BW / `gfx936` 设备为例，硬件配置如下：

| 资源 | 配置 | 调优含义 |
| --- | --- | --- |
| 计算单元 | 80 个 CU，每 CU 4 个 SIMD 单元 | 网格应提供足够的 program，避免小矩阵只占用少数 CU。 |
| Wavefront | 64 lane | `num_warps` 与分块大小应联合选择；一个 wavefront 由 64 个线程组成。 |
| Wave 驻留上限 | 每 SIMD 最多 10 个 wave | 每 CU 的理论上限是 40 个 wave；寄存器、LDS 和 workgroup 限制会使实际驻留数更低。 |
| 寄存器文件 | 每 SIMD 192 KB | 大分块与更多存活值会增加寄存器压力，降低可驻留 wave 数。 |
| LDS | 每 CU 64 KiB | workgroup 间共享容量；流水线缓冲与布局转换须计入实际用量。 |

该设备还支持 packed FP32 向量运算，可在编译器生成相应 packed 指令时同时处理两路 FP32 数据。它与下面的 DUMMA FP32 矩阵乘加使用不同的指令；仅凭输入类型无法判断编译器会选择哪条路径。上述资源数字针对这一目标设备，不应套用到所有海光 DCU。

## DUMMA 矩阵指令形状

《DTK 26.04.1 DUMMA 使用手册》第 3 节表 3.1 列出 `du::dumma::du_mma_sync` 支持的矩阵乘加形状；本机 DTK 26.04 的 `du_mma.h`/`du_mma.hpp` 也声明了相应接口。单个 wavefront 协同计算 `D = A × B + C` 的 `M × N` 输出分块。这里的 `16 × 16 × K` 是**单次 DUMMA 矩阵乘加**的形状，Triton 的 `BLOCK_M/N/K` 可以通过多次矩阵乘加与 K 维循环组成更大的分块。

| A、B 输入 | 累加器 | 形状 `M × N × K` | 手册列出的架构 |
| --- | --- | --- | --- |
| FP32 | FP32 | `16 × 16 × 4` | `gfx926/928/936/938` |
| FP32、TF32 | FP32 | `16 × 16 × 8` | `gfx928/936/938` |
| FP16、BF16 | FP32 | `16 × 16 × 16` | `gfx928/936/938` |
| 有符号/无符号 INT8 | INT32 | `16 × 16 × 32` | `gfx928/936/938` |
| 有符号/无符号 INT4 | INT32 | `16 × 16 × 64` | `gfx936/938` |
| FP64 | FP64 | `16 × 16 × 4` | `gfx926/936/938` |
| FP8（E4M3 或 E5M2） | FP32 | `16 × 16 × 32` | `gfx938` |

TF32 输入需按手册要求转换精度；INT4 输入需将两个 4 bit 值打包进一个字节，DTK 头文件将其类型放在实验性接口中。输入位宽降低时，单次乘加可沿 K 维处理更多元素。选择矩阵分块时，还应考虑布局、尾块 mask 与累加精度；`tl.dot` 是否编译成 DUMMA 需检查目标设备上的汇编。packed FP32 向量指令不属于此表。

## 软件流水线：平衡重叠与资源占用

对沿 K 维循环的矩阵乘法，可以比较不同 `num_stages`，让编译器尝试重叠数据加载与计算。流水线可能增加同时存活的数据，带来更多寄存器或 LDS 用量，进而减少驻留 workgroup；但不能假设每增加一个 stage 就固定增加一份 LDS 缓冲区。实际是否使用异步加载、分配多少 LDS，以生成代码和编译器报告为准。

估算 FP16 矩阵乘法单轮 A、B 分块的**数据量**时，可用 `(BLOCK_M + BLOCK_N) × BLOCK_K × 2` 字节。例如 `64 × 64 × 32` 对应 8 KiB 输入数据；这个数值不等于编译后 LDS 占用，因为布局转换、临时值和流水线方式也会影响资源。在 64 KiB LDS 的 CU 上，一个 workgroup 若实际占用 32 KiB LDS，仅从 LDS 容量考虑最多可同时驻留两个这样的 workgroup；若占用超过 32 KiB，则至多一个。寄存器与 wave 上限可能进一步降低驻留数。

## 分组调度：提高 L2 数据复用

矩阵乘法可以将二维输出分块展平到一维 grid，再通过 `GROUP_M` 把相邻 program 映射到一组 M 方向的分块。组内多个 M 分块处理相同的 N 分块时，可以复用 B 矩阵对应的 K×N 数据，减少从更低层存储重复读取。下面是这种分组映射的核心：

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

`group_size_m` 处理最后一个不足 `GROUP_M` 行的分组。执行顺序和缓存命中率仍受矩阵形状、CU 数量及编译器影响，因此要连同分块大小一起比较 `GROUP_M`，不能预设分组一定更快。

## 参考资料

- 《DCU 编程实战》（2026 版），第 6.8.4 节
- 《DTK 26.04.1 DUMMA 使用手册》，第 3 节表 3.1
- DTK 26.04 `du_mma.h`、`du_mma.hpp`（安装于 `/opt/dtk/include/`）
