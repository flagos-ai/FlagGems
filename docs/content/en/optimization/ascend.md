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

# Ascend optimization

## Hardware architecture

The Cube path executes matrix multiplication, the Vector path handles elementwise work and reductions, and the Scalar path handles address calculations and loop control. MTE engines move data from global memory (GM, usually HBM) into on-chip buffers. Cube uses L1, L0A, L0B, and L0C; Vector uses the Unified Buffer (UB). **UB and L0 are distinct spaces; L1 is a Cube-side buffer, not the cross-core L2 cache.** A kernel combining `tl.dot` with a Vector epilogue may need intermediate transfers and synchronization.

The following on-chip capacities come from the Triton-Ascend/AscendNPU-IR references; values are in KiB. GM/HBM capacity, bandwidth, and L2 size depend on the actual device. There is no single latency figure for these levels across models.

| Space | IR name | 910B / 910_93 | 910_95 / 950 series | Main use |
| --- | --- | ---: | ---: | --- |
| GM | `gm` | Device-specific | Device-specific | Global data, possibly cached by L2 |
| L1 | `cbuf` | 512 KiB | 512 KiB | Cube input buffer; typically plan for 32 B alignment |
| L0A / L0B | `ca` / `cb` | 64 KiB each | 64 KiB each | Matrix A/B input buffers |
| L0C | `cc` | 128 KiB | 256 KiB | Matrix accumulation result; plan related transfers around 512 B alignment |
| UB | `ub` | 192 KiB | 248 KiB usable (256 KiB nominal, 8 KiB reserved) | Vector workspace; typically plan for 32 B alignment |
| BT / FP Buffer | — | 1 / 7 KiB | 1 / 7 KiB | Bias table / FixPipe intermediate buffer |

Typical Cube flow is `GM → L1 → L0A/L0B → L0C → GM`; Vector flow is `GM → UB → GM`. On the 910_95/950 family, FixPipe can also send L0C results directly to UB and UB data can return to L1. The corresponding mixed path on 910B/910_93 generally goes through GM. Confirm paths and synchronization on the target chip and compiler.

### Core counts and grid

Query core counts by device model and compute path. The reference specification lists **24 Cube / 48 Vector** for 910B1/B2 and **20 Cube / 40 Vector** for 910B3/B4; 910_95 counts vary by SKU. Do not add Cube and Vector counts to choose one kernel's grid. FlagGems obtains `num_vectorcore` from Triton device properties for Vector work; matrix kernels can query `num_aicore`.

```python
import torch
import torch_npu  # Registers torch.npu
import triton.runtime.driver as driver

device_index = torch.npu.current_device()
props = driver.active.utils.get_device_properties(device_index)
vector_cores = props["num_vectorcore"]
cube_cores = props["num_aicore"]
```

Ordinary Triton-Ascend launches use a tuple `(x,)`, `(x, y)`, or `(x, y, z)`, with at most three dimensions. In the default path, `coreDim = x * y * z` must not exceed **65535**; an overflow can report `coreDim ... can't be greater than UINT16_MAX`. A 1D grid usually makes it easier to match the chosen core type. A 2D/3D grid counts the same product toward `coreDim`; keeping that product near the physical core count is a tuning choice, not a grid syntax rule.

For large shapes, cap the grid at `min(triton.cdiv(N, BLOCK_SIZE), vector_cores)` and let each program process multiple blocks, stepping by `tl.num_programs(0)`:

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

Some toolchain versions can reduce a logical grid with `TRITON_ALL_BLOCKS_PARALLEL=1`. Consider it only when programs have no ordering dependency, atomics, or cross-core synchronization, and verify correctness first.

### Tile capacity and alignment

If the 910B's 192 KiB UB held just **one** tile, its byte capacity would be 98,304 fp16/bf16 elements or 49,152 fp32 elements. These are neither legal Triton `BLOCK_SIZE` limits nor practical matmul tile sizes: simultaneously live inputs, outputs, intermediates, padding, and multiple buffers all consume space, and the compiler may impose a smaller block limit. Apply the same accounting to the 910_95's 248 KiB usable UB. When compilation reports `ub overflow` (often in bits), reduce `BLOCK_*` or introduce a `BLOCK_SIZE_SUB` loop within each core.

Check trailing-axis and matrix-row alignment for the Cube path. A 512 B row corresponds to 256 fp16/bf16 or 128 fp32 elements; shorter rows may be padded and transfer more data, rather than always causing a hardware error. Candidate searches can start around `BLOCK_M/N=128/256` and `BLOCK_K=128/256`, while accounting for separate A/B/C buffers and dtype support on the target chip. The reference lists FP8 `tl.dot` as unsupported on A2/A3, with FP8 support on 910_95 limited to specific `tl.dot_scaled` paths; element size alone does not establish a usable tile size.

## Triton compiler and launch options

FlagGems selects the Ascend NPU backend (`device_name="npu"`) and its installed Triton/FlagTree extension. Pass supported options as keywords to `kernel[grid](...)` or through a `triton.Config`. Option availability and defaults depend on the installed Triton-Ascend version; check its `NPUOptions` definition before using an extension-specific option.

| Option | What to tune | When to consider it |
| --- | --- | --- |
| `BLOCK_SIZE`, `BLOCK_M/N/K` | Compile-time tile sizes (`tl.constexpr`), not backend options | Balance core occupancy, UB/L1 use, and masked tails. |
| `num_warps` | Compiler parallelization/layout choice | Measure with the chosen tile; its SIMD meaning differs from GPU warp scheduling. |
| `num_stages` | Compiler pipeline setting | Confirm the installed compiler's behavior; FlagGems includes Ascend configurations with several stage values. |
| `compile_mode` | Selects the compiler path (`simd`, `unstructured_in_simt`, or `simt_only` where available) | Start with the installed compiler's default; test SIMT only for a suitable workload and verify output. |
| `multibuffer` | Overlap data movement and computation | Test looped kernels after checking local-buffer use; its default can differ by architecture. |
| `enable_flatten` | Flatten eligible loop structure | Try for Vector kernels, then verify generated code and performance. |
| `enable_mixed_cv`, `sync_solver`, `enable_auto_bind_sub_block` | Cube/Vector cooperation and synchronization | Consider together for kernels that genuinely combine the two paths; validate correctness first. |
| `enable_fp_fusion` | Floating-point fusion | Recheck numerical tolerance when changing it. |

For example, an elementwise kernel can compare `kernel[grid](..., BLOCK_SIZE=1024, num_warps=4)` against neighboring tile sizes. On a looped kernel, compare `multibuffer=True` with the baseline. Do not apply Cube/Vector options to every kernel by default.

The reference compiler defaults `multibuffer` to off for the 910_95/950 family; pass `multibuffer=True` explicitly when testing it. Double buffering increases simultaneously live on-chip data. Two equally sized UB buffers would each have about half the available space before other overhead, but use the compiler's actual allocation report for sizing.

### Common performance issues

| Symptom | Check and response |
| --- | --- |
| High `aiv_scalar_ratio` | Inspect generated IR for scalar loops. i64 vector arithmetic and some integer comparisons may lower to scalar operations on 910B/910_95. Use int32 only when values fit; if converting comparisons to fp32, check its exact integer range. An int32 equality comparison need not always be converted. |
| Masked loads block transfer/compute overlap | Test `tl.load(..., care_padding=False)` only when padding values cannot affect subsequent computation, reduction, or an unmasked store. Otherwise preserve a correct fill value. |
| Strided memory access | Change indexing or tiling to coalesce accesses. `.contiguous()` adds a copy, so compare end-to-end cost before preprocessing. |
| L2 conflicts or poor reuse in large matmul | Compare ordinary, grouped, and diagonal mappings with profiling. `grouped_launch_diagonal` is available in reference implementations; avoid unsupported `tl.swizzle2d` on the Ascend path. |
| Low-precision accumulation errors | For fp16/bf16 input, check fp32 accumulators in reductions and matrix operations and validate output tolerance. |

### Explicit buffers and synchronization

Where the installed compiler supports manual Cube/Vector data movement, `triton.backends.ascend.language.cann.extension.core` exposes `ascend_address_space` (GM/L1/UB/L0A/L0B/L0C), `copy`, `fixpipe`, `sync_block_set/wait/all`, and `sub_vec_id/num`. The L0C→UB `fixpipe` path is limited to capable 910_95/950 devices. Pair synchronization events and use IDs allowed by the installed version. Check the extension API, device capability, and generated transfer path before adopting these operations.

## Optimization workflow

1. Classify the kernel as Vector, Cube, or mixed; establish correctness and latency for representative shapes.
2. Make adjacent `tl.load`/`tl.store` addresses contiguous where possible. Use masks for tail elements and verify alignment on the actual dtype and layout.
3. Sweep grid size and tiles. Keep all live operands, intermediates, and any extra buffers within the relevant on-chip memory budget; reduce the tile if compilation reports UB pressure.
4. For looped workloads, test overlap through `multibuffer`. For mixed Cube/Vector workloads, inspect synchronization and intermediate transfers before enabling mixed-path options.
5. Profile the device to distinguish memory movement, Vector/Cube utilization, and launch overhead. Recheck numerical results after changes to precision or fusion.

## References

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
