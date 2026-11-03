---
title: Huawei Ascend
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

# Huawei Ascend Backend Optimization Guide

## 1. Hardware Architecture & Memory Hierarchy

The Ascend AI Core features a heterogeneous compute pipeline: the **Cube path** executes matrix multiplications, the **Vector path** handles element-wise operations and reductions, and the **Scalar path** manages address calculations and loop control.
Data moves from global memory (GM, typically HBM) into on-chip buffers via the Memory Transfer Engine (MTE). The Cube unit exclusively accesses L1, L0A, L0B, and L0C buffers; the Vector unit operates out of the Unified Buffer (UB).

> [!IMPORTANT]
> **UB and L0 are completely separate physical memory spaces**; **L1 is a Cube-dedicated on-chip buffer**, NOT a cross-core shared L2 cache. Composite kernels combining `tl.dot` with a Vector epilogue typically require intermediate data transfers and cross-path synchronization.

The table below lists the on-chip buffer capacities (in KiB) from the Triton-Ascend / AscendNPU-IR specifications across representative architectures. GM/HBM capacity, bandwidth, and L2 cache sizes vary by hardware model and SKU:

| Memory Space | IR Identifier | 910B / 910_93 | 910_95 / 950 Series | Primary Purpose & Alignment |
| :----------- | :------------ | :------------ | :------------------ | :-------------------------- |
| GM | `gm` | Device-specific | Device-specific | Global memory, cached via L2 |
| L1 | `cbuf` | 512 KiB | 512 KiB | Cube input buffer; plan for 32 B alignment |
| L0A / L0B | `ca` / `cb` | 64 KiB each | 64 KiB each | Matrix A/B operand on-chip input buffers |
| L0C | `cc` | 128 KiB | 256 KiB | Matrix accumulation output; transfers plan for 512 B alignment |
| UB | `ub` | 192 KiB | 248 KiB usable (256 KiB nominal, 8 KiB reserved) | Vector workspace; plan for 32 B alignment |
| BT / FP Buffer | — | 1 / 7 KiB | 1 / 7 KiB | Bias table / FixPipe intermediate buffer |

- **Typical Cube Data Flow**: `GM → L1 → L0A/L0B → L0C → GM`
- **Typical Vector Data Flow**: `GM → UB → GM`

The 910_95/950 family further supports routing L0C accumulation results directly into UB via FixPipe and writing UB data back to L1; the corresponding mixed paths on 910B/910_93 chips generally route through global memory (GM).

### 1.1 Core Counts & Grid Mapping

Ascend physical core counts are queried separately by compute unit type. Typical specifications:
- **910B1 / 910B2**: 24 Cube / 48 Vector
- **910B3 / 910B4**: 20 Cube / 40 Vector
- **910_95 Series**: Configured dynamically by SKU

> [!WARNING]
> Do NOT sum the Cube and Vector core counts to configure the launch grid of a single kernel.

*FlagGems* queries `num_vectorcore` from Triton device properties for Vector kernels, and `num_aicore` for matrix kernels:

```python
import torch
import torch_npu  # Registers torch.npu
import triton.runtime.driver as driver

device_index = torch.npu.current_device()
props = driver.active.utils.get_device_properties(device_index)
vector_cores = props["num_vectorcore"]
cube_cores = props["num_aicore"]
```

In Triton-Ascend, launch grids use 1D, 2D, or 3D tuples. Under the default pipeline, `coreDim = x * y * z` must not exceed **65535** (otherwise a `coreDim ... can't be greater than UINT16_MAX` error is triggered).

When processing large tensor shapes, limit the grid to the physical core count and employ a grid-stride loop inside each program stepping by `tl.num_programs(0)`:

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

### 1.2 Tile Capacity & Memory Alignment

- **UB Budgeting**: If a single tile occupied the 910B's 192 KiB UB, its theoretical capacity would be 98,304 FP16/BF16 elements or 49,152 FP32 elements. However, in practice, simultaneously live inputs, outputs, temporaries, alignment padding, and multibuffering all consume space. If a `ub overflow` compilation error occurs (often reported in bits), reduce `BLOCK_*` or use an internal `BLOCK_SIZE_SUB` loop.
- **Cube Alignment**: The Cube path requires careful attention to row-width and trailing-axis alignment. A 512 B row corresponds to 256 FP16/BF16 or 128 FP32 elements; unaligned dimensions may trigger compiler padding and redundant memory transfers.

### 1.3 910C and 950DT Hardware Specifications

> [!WARNING]
> Core counts, frequencies, and memory configurations vary across SKUs. Always verify against device-specific configurations via `platform_config/<SoC_version>.ini` (or using `acl.get_soc_name()` / `npu-smi`) rather than inferring them from chip family names. The tables below summarize common SKU configurations found in CANN `platform_config`.

**Ascend910C** (A3 architecture, dual-die package, categorized under `ascend910b_list` in CANN, SIMD-only) includes six common `Ascend910_93xx` SKUs. All share `ai_cpu_cnt=6` and `l1_size=512 KiB`, while HBM capacity and L2 cache scale with the Vector core count:

| SoC Version | AI Core (Cube) | Vector Core | Cube Clock | HBM Capacity | L2 Cache | UB / AI Core |
| :---------- | :------------- | :---------- | :--------- | :----------- | :------- | :----------- |
| `Ascend910_9362` | 20 | 40 | 1.5 GHz | 32 GiB | 168 MiB | 192 KiB |
| `Ascend910_9372` | 20 | 40 | 1.8 GHz | 64 GiB | 192 MiB | 192 KiB |
| `Ascend910_9381` | 24 | 48 | 1.8 GHz | 64 GiB | 192 MiB | 192 KiB |
| `Ascend910_9382` | 24 | 48 | 1.8 GHz | 64 GiB | 192 MiB | 192 KiB |
| `Ascend910_9391` | 24 | 48 | 1.85 GHz | 64 GiB | 192 MiB | 192 KiB |
| `Ascend910_9392` | 24 | 48 | 1.85 GHz | 64 GiB | 192 MiB | 192 KiB |

- `Ascend910_9362` / `_9372`: 20 Cube cores; `_9362` has a smaller 32 GiB HBM capacity.
- `Ascend910_9381` / `_9382`: 24 Cube cores at 1.8 GHz with identical configurations; `_9382` is the most common SKU in production.
- `Ascend910_9391` / `_9392`: 24 Cube cores with an increased clock of 1.85 GHz.

**Ascend950DT** (A5 architecture, decode-oriented, sharing silicon with 950PR with up to 36 AI subsystems). In CANN 9.0.0/9.1.0, the `Ascend950DT_9582` configuration (`platform_config/Ascend950DT_9582.ini`) reports:

| Parameter | Value | Description |
| :-------- | :---- | :---------- |
| `cube_core_cnt` / `vector_core_cnt` | 32 / 64 | Cube / Vector core count |
| `l1_size` | 512 KiB | L1 cache size |
| `l2_size` | 128 MiB | L2 cache size |
| `ub_size` | 248 KiB (253,952 B) | Usable Unified Buffer size |
| `cube_freq` / `vec_freq` | 1.65 GHz | Cube / Vector operating frequency |
| `memory_size` | 96 GiB | HBM capacity |

#### Ascend950DT_9582 Theoretical Peak vs. Measured Performance

`Ascend950DT_9582` corresponds to the 32-Cube-core tier. The table below compares the theoretical peak performance against measured single-node throughput for `Ascend950DT_9582` (measured using CANN 9.1.0 and `torch_npu`):

| Metric | Theoretical Peak | Measured | Benchmark / Notes |
| :----- | :--------------- | :------- | :---------------- |
| `matmul` (Cube+Vector BF16 / FP16) | 486 TFLOP/s | 430.4 TFLOP/s | `torch.matmul`, 8192³, median of 30 runs |
| `matmul` (Cube+Vector TF32) | 243 TFLOP/s | 27.0 TFLOP/s | FP32 without HF32 path, 8192³ |
| `matmul` (Cube+Vector FP8 / HiF8 / MXFP8) | 919 TFLOP/s | — | — |
| `matmul` (Cube+Vector MXFP4) | 1784 TFLOP/s | — | — |
| `matmul` (Cube+Vector INT8) | 919 TOPS | — | — |
| Memory Bandwidth | 4 TB/s | ≈2.7–3.1 TB/s | ≥1024 MiB copy / triad / read (large buffers) |
| Memory Capacity | 96 GB | 96 GiB | `platform_config/Ascend950DT_9582.ini` (`memory_size`) |
| `conv2d` FP16 | — | 353–364 TFLOP/s | `F.conv2d`, large 1x1 convolutions |
| `conv2d` FP32 (HF32 on / off) | — | 176–183 / 26.9 TFLOP/s | `torch.npu.conv.allow_hf32` |

> [!NOTE]
> 1. **Data Source**: Theoretical peaks are derived from the *Ascend 950 NPU Architecture Whitepaper* (p13, Table 3-1, 950DT 32-core configuration).
> 2. **Measurement Scope**: Measured results depend on host environment, drivers, and software stack versions; they serve as empirical benchmarks rather than strict hardware ceilings.
> 3. **Bandwidth Profiling**: Small buffer transfers (e.g., 64 MiB copy) can show inflated bandwidth (~1.8× higher) due to cache effects; large buffer transfers (≥1024 MiB) should be used to measure sustained memory bandwidth.
> 4. **Operator Differences**: Different operators (e.g., `conv2d` vs. `matmul`) utilize hardware pipelines and execution units differently; their saturation throughputs should not be directly equated.

## 2. Triton Compiler & Launch Options

*FlagGems* designates Ascend as an NPU backend (`device_name="npu"`), utilizing the installed Triton-Ascend / FlagTree extension. Options can be passed as keyword arguments to `kernel[grid](...)` or through a `triton.Config`:

| Option | Target | Tuning Recommendations |
| :----- | :----- | :--------------------- |
| `BLOCK_SIZE`, `BLOCK_M/N/K` | Compile-time tile sizes (`tl.constexpr`) | Balance core occupancy, UB/L1 buffer budget, and masked tail overhead |
| `num_warps` | Compiler parallelization & layout choice | Measure alongside tile sizes; SIMD semantics differ from GPU warp scheduling |
| `num_stages` | Software pipeline depth | Regulates asynchronous prefetching; confirm with your installed compiler version |
| `compile_mode` | Compiler path selection (`simd` / `simt_only` etc.) | Default to `simd`; experiment with SIMT only for suitable workloads after verifying accuracy |
| `multibuffer` | Overlap data movement and compute | Test on looped kernels after checking local buffer budgets; watch for doubled UB usage |
| `enable_flatten` | Loop structure flattening | Useful for simple Vector kernels to eliminate branching overhead |
| `enable_mixed_cv`, `sync_solver` | Cube & Vector synchronization | Enable only for composite kernels that combine matrix multiplication and vector ops |
| `enable_fp_fusion` | Floating-point operation fusion | Re-verify numerical tolerances when toggling this option |

> [!NOTE]
> The reference compiler defaults `multibuffer` to disabled for the 910_95/950 family. When exploring double buffering, pass `multibuffer=True` explicitly. Be aware that double buffering almost doubles live on-chip memory usage.

## 3. Common Performance Bottlenecks & Troubleshooting

| Performance Issue | Diagnosis & Mitigation |
| :---------------- | :--------------------- |
| **High `aiv_scalar_ratio`** | Check generated IR for scalar loops. On 910B, `int64` vector math and integer comparisons may degrade to scalar mode; cast to `int32` where values fit in range. |
| **Mask load blocks compute overlap** | Only test `tl.load(..., care_padding=False)` if padding data will **never** participate in reductions or unmasked stores; otherwise keep standard masked loads. |
| **Non-contiguous strided access** | Restructure tensor indices or tiling to enforce coalesced memory access. Avoid unnecessary `.contiguous()` copies which often erase kernel gains. |
| **L2 cache thrashing on large matrices** | Implement grouped scheduling (Grouped Launch); avoid directly calling unsupported `tl.swizzle2d` on Ascend. |
| **Low-precision accumulation overflow** | For FP16/BF16 inputs, ensure reductions and matrix multiplication accumulate into FP32 accumulators to prevent numerical drift. |

## 4. Recommended Optimization Workflow

1. **Classify the Compute Path**: Determine whether the kernel is purely Vector, purely Cube, or mixed; establish accuracy and latency baselines with representative shapes.
2. **Ensure Coalesced Access**: Align strides and pointer offsets to maximize memory bandwidth utilization.
3. **Budget On-chip Memory**: Fit tiles strictly within UB (192 KiB / 248 KiB) and L1 (512 KiB) physical limits to prevent register spills and UB overflows.
4. **Explore Overlap & Pipelining**: For looped kernels, evaluate `multibuffer=True` and tune `num_stages`.
5. **Profile on Device**: Use the Ascend Profiler to inspect Cube vs. Vector MTE utilization, validating that latency speedups pass strict numerical verification gates.

## 5. References

- [cannbot-knowledge: NPU compiler options](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/compile_params.md)
- [cannbot-knowledge: performance overview](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/perf_optimization_overview.md)
- [cannbot-knowledge: tiling](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/tiling.md)
- [cannbot-knowledge: grid and program IDs](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/grid_and_program_id.md)
- [cannbot-knowledge: hardware specification IR mapping](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/apis/ir/reference/hardware_specs_ir.md)
- [cannbot-knowledge: memory hierarchy](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/apis/ir/architecture/memory_hierarchy.md)
- [cannbot-knowledge: scalar lowering](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/scalar_degradation_avoidance.md)
- [cannbot-knowledge: `care_padding` conditions](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/care_padding.md)
- [cannbot-knowledge: NPU dtypes and alignment](https://gitcode.com/cann/cannbot-knowledge/blob/2dc417d1e7419f7b6f2f9a926860876ef8ae0887/knowledge/ops/triton/optimizations/techniques/data_types_ascend.md)
- [cannbot-knowledge: Ascend950DT measured observations](https://gitcode.com/cann/cannbot-knowledge/blob/master/knowledge/common/platforms/concepts/ascend950dt_measured_observations.md)
- [cannbot-knowledge: Ascend950DT target platform](https://gitcode.com/cann/cannbot-knowledge/blob/master/knowledge/common/platforms/concepts/target_ascend950dt.md)
- [cannbot-knowledge: Ascend950DT vs 950PR SKU overrides](https://gitcode.com/cann/cannbot-knowledge/blob/master/knowledge/common/platforms/concepts/ascend950dt_vs_pr_overrides.md)
- [Huawei Ascend 950 NPU Architecture Whitepaper (PDF, Chinese)](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)
- [cannbot-skills: Ascend compilation options](https://gitcode.com/cann/cannbot-skills/blob/affd5a88dd956022a598adf2f5ed9f3e26718d48/ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/04-Compilation-Pipeline/07-compile-options.md)
- [cannbot-skills: Ascend memory model and extension API](https://gitcode.com/cann/cannbot-skills/blob/affd5a88dd956022a598adf2f5ed9f3e26718d48/ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/01-Programming-Model/03-memory-model.md)
- [Triton-Ascend backend compiler (`NPUOptions`)](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/backend/compiler.py)
- [Triton-Ascend: environment variables and compiler options reference](https://triton-ascend.readthedocs.io/en/v3.6.0/environment_variable_and_compiler_options_reference.html#environment-variables-and-compiler-options)
