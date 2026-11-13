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

import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .sort import sort_stable

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mode_kernel(
    sorted_inp,
    sorted_indices,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    WIDEN_VALUE: tl.constexpr,
):
    # Run-length scan over ascending-sorted rows: the longest run of equal
    # values is the mode, and because the rows are sorted ascending the first
    # run that reaches the best count is the smallest tied value, matching
    # torch.mode's tie-break. The reported index only has to point at an
    # occurrence of the winning value.
    #
    # gcu400: int8/uint8 value lanes never reach instruction selection (LLVM
    # "Cannot select: v512i1 = concat_vectors" on the narrow compare/select
    # chain; reproduces for every BLOCK_M/num_warps of the naive_reduction
    # sweep, while int16/int32/int64/float lanes are fine). Widening the value
    # to int32 right after the load dodges the selector gap; the cast is exact
    # for the full int8/uint8 range. All other dtypes keep their native lanes
    # (preserving exact `==` semantics, e.g. -0.0 == +0.0 for floats).
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offset < M

    # Load first element
    n0 = tl.arange(0, 1)
    offset0 = m_offset[:, None] * N + n0[None, :]
    mask0 = mask_m[:, None] & (n0[None, :] < N)
    val0 = tl.load(sorted_inp + offset0, mask=mask0, other=0)
    idx0 = tl.load(sorted_indices + offset0, mask=mask0, other=0)
    if WIDEN_VALUE:
        val0 = val0.to(tl.int32)

    # Squeeze the n-dimension (size 1)
    cur_value = val0.reshape([BLOCK_M])
    cur_index = idx0.reshape([BLOCK_M])
    cur_count = tl.full([BLOCK_M], 1, dtype=tl.int64)
    best_value = cur_value
    best_index = cur_index
    best_count = tl.full([BLOCK_M], 1, dtype=tl.int64)

    # Scan remaining elements one by one
    for i in range(1, N):
        ni = tl.full([1], i, dtype=tl.int32)
        offset_i = m_offset[:, None] * N + ni[None, :]
        mask_i = mask_m[:, None] & (ni[None, :] < N)
        val_i = tl.load(sorted_inp + offset_i, mask=mask_i, other=0)
        idx_i = tl.load(sorted_indices + offset_i, mask=mask_i, other=0)
        val_i = val_i.reshape([BLOCK_M])
        idx_i = idx_i.reshape([BLOCK_M])
        if WIDEN_VALUE:
            val_i = val_i.to(tl.int32)

        same = val_i == cur_value
        cur_count = tl.where(same, cur_count + 1, tl.full([BLOCK_M], 1, dtype=tl.int64))
        cur_value = tl.where(same, cur_value, val_i)
        cur_index = idx_i  # Any index inside the winning run is a valid answer

        better = cur_count > best_count
        best_count = tl.where(better, cur_count, best_count)
        best_value = tl.where(better, cur_value, best_value)
        best_index = tl.where(better, cur_index, best_index)

    if WIDEN_VALUE:
        best_value = best_value.to(out_value.dtype.element_ty)
    tl.store(out_value + m_offset, best_value, mask=mask_m)
    tl.store(out_index + m_offset, best_index, mask=mask_m)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS MODE")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    shape = list(inp.shape)
    N = shape[dim]
    M = inp.numel() // N

    # gcu400: sort through the local stable radix (sort_stable) instead of the
    # upstream mode-private radix. The private variant histogram kernel uses
    # tl.atomic_add + a decoupled-lookback sweep, keeps a torch.cumsum on the
    # host side, and launches grid.x = m * grid_n programs, which exceeds the
    # 65535 grid.x hardware limit as soon as the row count reaches 65536
    # (test_mode_small_rows_launch_boundary / byte histogram 65536 rows).
    # sort_stable is atomic-free, lookback-free, caps every grid dimension for
    # any row count, and does its prefix sums inside its own kernels.
    if dim != inp.ndim - 1:
        work = torch.movedim(inp, dim, -1).contiguous()
    else:
        work = inp.contiguous()
    sorted_inp, sorted_indices = sort_stable(work, stable=True, dim=-1)

    sorted_flat = sorted_inp.reshape(M, N)
    indices_flat = sorted_indices.reshape(M, N)

    out_value = torch.empty(M, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(M, dtype=torch.int64, device=inp.device)

    if M:
        widen_value = inp.dtype in (torch.int8, torch.uint8)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        with torch_device_fn.device(inp.device):
            mode_kernel[grid](
                sorted_flat,
                indices_flat,
                out_value,
                out_index,
                M,
                N,
                BLOCK_M,
                widen_value,
                num_warps=4,
            )

    out_shape = list(shape)
    out_shape[dim] = 1
    out_value = out_value.reshape(out_shape)
    out_index = out_index.reshape(out_shape)

    if not keepdim:
        out_value = out_value.squeeze(dim)
        out_index = out_index.squeeze(dim)

    Mode_out = namedtuple("mode", ["values", "indices"])
    return Mode_out(values=out_value, indices=out_index)
