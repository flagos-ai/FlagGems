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

# Hygon optimization

## Hardware architecture

FlagGems runs Hygon DCU kernels through a HIP-compatible Triton backend. The execution model groups threads into warps and workgroups on compute units. A kernel reads device memory, may stage reused values in on-chip shared memory (LDS), and uses registers for thread-local values. Occupancy depends on the registers and LDS used by each workgroup, as well as the number of warps. These limits vary across DCU models and compiler versions. FlagGems's `gfx936` tuning notes describe 64-lane warps and 64 KiB LDS; the current Hygon attention implementation filters configurations against a 64 KiB shared-memory budget. Check the limits of other devices separately.

The backend descriptor uses `device_name="cuda"` for PyTorch dispatch and `triton_extra_name="hip"` for the Triton path. The device query command is `hy-smi`. These are implementation details to check when diagnosing dispatch or compilation.

## Triton compiler and launch options

The following are candidates for `kernel[grid](..., ...)` or `triton.Config(...)`. They are compilation or launch settings; `BLOCK_*` values are kernel `tl.constexpr` arguments. FlagGems's Hygon tuning table searches several combinations rather than prescribing one setting for all shapes.

| Option | Effect | Tuning check |
| --- | --- | --- |
| `BLOCK_SIZE`, `BLOCK_M/N/K` | Work and data per program | Increase reuse and memory coalescing without exhausting registers or LDS. |
| `num_warps` | Warps assigned to a program | Compare values supported by the installed compiler; Hygon configurations commonly test 4 and 8, and some reductions use 16. |
| `num_stages` | Software pipeline depth for eligible loops | Compare 1–3 with the baseline where loads and computation can overlap; more stages may consume more LDS. |
| `enable_fp_fusion` | Allows floating-point fusion | Check numerical tolerances after changing it. |
| `waves_per_eu` | HIP backend occupancy hint, where supported | Use only if the installed Triton/DTK compiler exposes it; measure rather than assuming a universal value. |

For example, a matrix kernel can compare `triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2)` with other legal tile/warp/stage combinations. A larger tile or deeper pipeline can lose performance when it reduces resident workgroups.

## Optimization workflow

1. Benchmark the actual shapes, dtypes, and layouts. Separate small, launch-limited inputs from large, bandwidth- or compute-limited inputs.
2. Make adjacent lanes access adjacent elements when the layout permits. Mask tails and avoid unnecessary global-memory round trips by reusing a tile within a program.
3. Sweep tiles and `num_warps` together. If compilation reports shared-memory overflow or occupancy falls, reduce tile dimensions or `num_stages`.
4. For reductions and GEMV, compare a vectorized reduction with `tl.dot` on the target compiler. FlagGems uses a vectorized reduction in its Hygon `mv` path because the current `gfx936` lowering of `tl.dot` adds shared-memory movement for that case.
5. Keep the fastest configuration only after checking numerical correctness and repeatable latency. Use [pre-tuning](/FlagGems/usage/tuning/) to populate the persistent tuning cache for production shapes.

## References

- [FlagGems Hygon backend descriptor](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/__init__.py)
- [FlagGems Hygon tuning configurations](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/tune_configs.yaml)
- [FlagGems Hygon attention shared-memory filter](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/ops/attention.py)
- [FlagGems Hygon `gfx936` tuning notes](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/ops/addmm_.py)
- [FlagGems Hygon GEMV implementation](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/_hygon/ops/mv.py)
- [Triton `Config` API](https://triton-lang.org/main/python-api/generated/triton.Config.html)
