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

## 硬件架构

FlagGems 通过兼容 HIP 的 Triton 后端运行海光 DCU kernel。执行时，线程按 warp 和 workgroup 组织到计算单元上。kernel 从设备内存读取数据，可将复用的数据放入片上共享内存（LDS），并用寄存器保存线程局部值。每个 workgroup 占用的寄存器、LDS 和 warp 数共同影响驻留并行度。这些限制随 DCU 型号和编译器版本变化。FlagGems 的 `gfx936` 调优注释记载了 64 线程 warp 和 64 KiB LDS；当前 Hygon attention 实现也按 64 KiB 共享内存预算筛选配置。其他设备应单独核对资源限制。

后端描述符用 `device_name="cuda"` 进行 PyTorch 分发，用 `triton_extra_name="hip"` 选择 Triton 路径，并以 `hy-smi` 查询设备。排查分发或编译问题时可核对这些设置。

## Triton 编译与启动参数

下表参数可作为 `kernel[grid](..., ...)` 或 `triton.Config(...)` 的候选配置。其中 `BLOCK_*` 是 kernel 的 `tl.constexpr` 参数。FlagGems 的 Hygon 调优表搜索多组配置，并未为所有形状指定一套固定值。

| 参数 | 作用 | 调优检查 |
| --- | --- | --- |
| `BLOCK_SIZE`、`BLOCK_M/N/K` | 每个 program 的工作量与数据量 | 增加复用和连续访问，同时避免寄存器或 LDS 超额。 |
| `num_warps` | 每个 program 使用的 warp 数 | 比较已安装编译器支持的值；Hygon 配置常搜索 4、8，部分归约使用 16。 |
| `num_stages` | 适用循环的软件流水深度 | 当加载与计算可重叠时，将 1–3 与基线比较；更多 stage 可能增加 LDS 占用。 |
| `enable_fp_fusion` | 浮点融合 | 调整后检查数值容差。 |
| `waves_per_eu` | HIP 后端支持时的驻留并行度提示 | 仅在安装的 Triton/DTK 编译器提供该参数时使用，并通过实测选值。 |

例如，矩阵 kernel 可比较 `triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2)` 与其他合法的分块、warp 和 stage 组合。更大的分块或更深的流水线若减少驻留 workgroup，性能可能下降。

## 优化步骤

1. 对实际形状、数据类型和布局建立基线，区分小输入的启动开销与大输入的访存或计算瓶颈。
2. 布局允许时，让相邻 lane 访问相邻元素。尾块加 mask，并在单个 program 内复用分块，减少往返设备内存的次数。
3. 联合搜索分块与 `num_warps`。遇到共享内存超额或驻留并行度下降时，减小分块或 `num_stages`。
4. 对归约和 GEMV，在目标编译器上比较向量归约与 `tl.dot`。FlagGems 的 Hygon `mv` 使用向量归约，因为当前 `gfx936` 下该场景的 `tl.dot` 降低过程引入了额外共享内存搬运。
5. 验证数值正确性与稳定耗时后再保留最快配置。可用[预调优](/FlagGems/zh-cn/usage/tuning/)为生产形状填充持久化调优缓存。

## 参考资料

- [FlagGems Hygon 后端描述符](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/__init__.py)
- [FlagGems Hygon 调优配置](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/tune_configs.yaml)
- [FlagGems Hygon attention 共享内存筛选](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/ops/attention.py)
- [FlagGems Hygon `gfx936` 调优注释](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/ops/addmm_.py)
- [FlagGems Hygon GEMV 实现](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/ops/mv.py)
- [Triton `Config` API](https://triton-lang.org/main/python-api/generated/triton.Config.html)
