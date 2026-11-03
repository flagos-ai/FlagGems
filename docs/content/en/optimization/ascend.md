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
- [cannbot-skills: Ascend compilation options](https://gitcode.com/cann/cannbot-skills/blob/affd5a88dd956022a598adf2f5ed9f3e26718d48/ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/04-Compilation-Pipeline/07-compile-options.md)
- [cannbot-skills: Ascend memory model and extension API](https://gitcode.com/cann/cannbot-skills/blob/affd5a88dd956022a598adf2f5ed9f3e26718d48/ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/01-Programming-Model/03-memory-model.md)
- [Triton-Ascend backend compiler (`NPUOptions`)](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/backend/compiler.py)
- [Triton-Ascend: environment variables and compiler options reference](https://triton-ascend.readthedocs.io/en/v3.6.0/environment_variable_and_compiler_options_reference.html#environment-variables-and-compiler-options)
