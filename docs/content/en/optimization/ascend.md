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

Ascend AI Cores expose distinct **Cube** matrix and **Vector** compute paths. Matrix tiles use the Cube path and its L1/L0 buffers; elementwise operations and reductions use the Vector path and its Unified Buffer (UB). Global memory transfers and synchronization connect these paths. Available core counts and on-chip capacity vary by device generation, so query the target device before setting a grid or a tile budget. A kernel that combines `tl.dot` with a Vector epilogue may need both paths and additional intermediate storage.

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
- [cannbot-skills: Ascend compilation options](https://gitcode.com/cann/cannbot-skills/blob/affd5a88dd956022a598adf2f5ed9f3e26718d48/ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/04-Compilation-Pipeline/07-compile-options.md)
- [Triton-Ascend backend compiler (`NPUOptions`)](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/backend/compiler.py)
