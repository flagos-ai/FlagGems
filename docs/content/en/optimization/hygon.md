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

Triton kernels run on Hygon DCUs through the HIP path. Tile sizes, warp counts, pipeline depth, and program ordering determine register, on-chip shared-memory (LDS), and cache use. This guide uses Hygon BW / `gfx936` to explain optimization mechanisms tied directly to those resources.

## Hardware and configuration limits

The Hygon BW / `gfx936` device discussed here has the following configuration:

| Resource | Configuration | Tuning implication |
| --- | --- | --- |
| Compute units | 80 CUs, with 4 SIMD units per CU | Launch enough programs to use the CUs, especially for small matrices. |
| Wavefront | 64 lanes | Choose `num_warps` with the tile size; one wavefront contains 64 threads. |
| Resident-wave limit | Up to 10 waves per SIMD | The theoretical CU limit is 40 waves; registers, LDS, and workgroup limits can lower actual residency. |
| Register file | 192 KB per SIMD | Larger tiles and more live values can reduce resident waves. |
| LDS | 64 KiB per CU | Pipeline buffers and layout conversions must fit within the shared capacity. |

This device also supports packed FP32 vector operations, which can process two FP32 values per packed instruction when the compiler emits one. These use different instructions from FP32 DUMMA matrix multiplication below; input dtype alone does not establish which path the compiler chooses. The resource figures above apply to this target device, not every Hygon DCU.

## DUMMA matrix instruction shapes

Table 3.1 in Section 3 of the *DTK 26.04.1 DUMMA User Manual* lists the matrix multiply-accumulate shapes available through `du::dumma::du_mma_sync`. The installed DTK 26.04 `du_mma.h`/`du_mma.hpp` headers declare the corresponding interfaces. One wavefront cooperatively computes an `M × N` output tile for `D = A × B + C`. The `16 × 16 × K` entries below describe **one DUMMA matrix multiply-accumulate operation**; Triton `BLOCK_M/N/K` tiles can combine multiple operations and K-loop iterations.

| A/B input | Accumulator | Shape `M × N × K` | Architectures in the manual |
| --- | --- | --- | --- |
| FP32 | FP32 | `16 × 16 × 4` | `gfx926/928/936/938` |
| FP32, TF32 | FP32 | `16 × 16 × 8` | `gfx928/936/938` |
| FP16, BF16 | FP32 | `16 × 16 × 16` | `gfx928/936/938` |
| Signed/unsigned INT8 | INT32 | `16 × 16 × 32` | `gfx928/936/938` |
| Signed/unsigned INT4 | INT32 | `16 × 16 × 64` | `gfx936/938` |
| FP64 | FP64 | `16 × 16 × 4` | `gfx926/936/938` |
| FP8 (E4M3 or E5M2) | FP32 | `16 × 16 × 32` | `gfx938` |

TF32 inputs need the precision conversion described in the manual. INT4 inputs must pack two 4-bit values into a byte; the DTK headers expose their types through an experimental interface. Lower input precision allows one operation to cover more K elements. Choose matrix tiles with layout, tail masks, and accumulator precision in mind. Inspect target-device assembly to confirm whether `tl.dot` lowers to DUMMA. Packed FP32 vector instructions are outside this table.

## Software pipelining: balance overlap and resource use

For matrix multiplication with a loop over K, compare `num_stages` values so the compiler can attempt to overlap loads and computation. Pipelining can keep more data live, increasing register or LDS use and reducing resident workgroups. An additional stage does not necessarily allocate one extra LDS buffer. Inspect the generated code and compiler resource report to see whether asynchronous loads are used and how much LDS is actually allocated.

For an FP16 matrix multiplication, the **input data size** of one A and B tile pair is approximately `(BLOCK_M + BLOCK_N) × BLOCK_K × 2` bytes. With `64 × 64 × 32`, that is 8 KiB. It is not the compiled LDS footprint: layout conversions, temporaries, and pipelining also affect resource use. On a CU with 64 KiB LDS, a workgroup that actually uses 32 KiB permits at most two resident workgroups based on LDS capacity alone; above 32 KiB, at most one. Register and wave limits may reduce residency further.

## Grouped scheduling: improve L2 data reuse

Matrix multiplication can flatten the two-dimensional output-tile grid to one dimension, then use `GROUP_M` to map neighboring programs to a group of M tiles. When several M tiles in a group process the same N tile, they can reuse the corresponding K×N tile of matrix B in L2 and reduce repeated reads from lower levels of memory. The core grouped mapping is:

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

`group_size_m` handles the final group when fewer than `GROUP_M` M tiles remain. Execution order and cache hit rate still depend on matrix shape, CU count, and compiler behavior. Compare `GROUP_M` along with the tile sizes rather than assuming grouping always helps.

## References

- 《DCU 编程实战》 (2026 edition), Section 6.8.4
- *DTK 26.04.1 DUMMA User Manual*, Section 3, Table 3.1
- DTK 26.04 `du_mma.h` and `du_mma.hpp` (installed under `/opt/dtk/include/`)
