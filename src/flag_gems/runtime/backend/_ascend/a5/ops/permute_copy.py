# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _tile_kernel(
    src_ptr,
    dst_ptr,
    CHUNK_D1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    SI: tl.constexpr,
    SJ: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TRANS: tl.constexpr,
    MASKED: tl.constexpr,
):
    c = tl.program_id(0)
    t0 = c // CHUNK_D1
    t1 = c - t0 * CHUNK_D1
    sbase = t0 * S0 + t1 * S1
    i = tl.program_id(1) * BM + tl.arange(0, BM)
    j = tl.program_id(2) * BN + tl.arange(0, BN)
    dptr = dst_ptr + c * (ROWS * COLS) + i[:, None] * COLS + j[None, :]
    if TRANS:
        # source is unit stride along the row axis: load (BN, BM) so the
        # innermost source dimension is contiguous, then transpose on chip.
        sptr = src_ptr + sbase + i[None, :] * SI + j[:, None] * SJ
        if MASKED:
            m0 = (i[None, :] < ROWS) & (j[:, None] < COLS)
            m1 = (i[:, None] < ROWS) & (j[None, :] < COLS)
            tl.store(dptr, tl.trans(tl.load(sptr, mask=m0, other=0)), mask=m1)
        else:
            tl.store(dptr, tl.trans(tl.load(sptr)))
    else:
        sptr = src_ptr + sbase + i[:, None] * SI + j[None, :] * SJ
        if MASKED:
            m = (i[:, None] < ROWS) & (j[None, :] < COLS)
            tl.store(dptr, tl.load(sptr, mask=m, other=0), mask=m)
        else:
            tl.store(dptr, tl.load(sptr))


def permute_copy(x, dims):
    rank = x.ndim
    if (
        x.device.type != "npu"
        or not 2 <= rank <= 4
        or not isinstance(dims, (list, tuple))
        or len(dims) != rank
        or any(not -rank <= d < rank for d in dims)
        or len({d % rank for d in dims}) != rank
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float32, torch.int32, torch.int64)
        or x.numel() >= 2**31
        or x.is_neg()
    ):
        from flag_gems.runtime.backend._ascend.ops.permute_copy import (
            permute_copy as ascend_permute_copy,
        )

        return ascend_permute_copy(x, dims)
    shape = tuple(x.shape[d] for d in dims)
    stride = tuple(x.stride(d) for d in dims)
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    if not out.numel():
        return out
    rows, cols = shape[-2:]
    bm = min(triton.next_power_of_2(rows), 64)
    bn = min(triton.next_power_of_2(cols), 64)
    with torch_device_fn.device(x.device):
        _tile_kernel[
            (math.prod(shape[:-2]), triton.cdiv(rows, bm), triton.cdiv(cols, bn))
        ](
            x,
            out,
            shape[-3] if rank >= 3 else 1,
            stride[-4] if rank == 4 else 0,
            stride[-3] if rank >= 3 else 0,
            stride[-2],
            stride[-1],
            rows,
            cols,
            bm,
            bn,
            stride[-2] == 1,
            rows % bm != 0 or cols % bn != 0,
        )
    return out
