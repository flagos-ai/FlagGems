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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn  # noqa: F401
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs). The generic pointwise_dynamic
# codegen launches cdiv(out_numel, BLOCK_SIZE) CTAs, which exceeds the
# 65535 grid.x hardware limit for large outputs (e.g. 16x7x57x32x29 ->
# >200k CTAs -> OutOfResources). A fixed grid keeps every launch within
# the limit.
_GRID_SIZE = 48


# Persistent-grid programs for the flat high-rank pad path.  grid.x must stay
# <= 65535 on GCU400; a strided loop covers the remaining tiles.
_MAX_PROGRAMS = 2048


@triton.jit(do_not_specialize=["value"])
def _pad_flat_kernel(
    in_ptr,
    out_ptr,
    value,
    N,
    out_shape_ptr,
    out_stride_ptr,
    in_stride_ptr,
    valid_start_ptr,
    valid_end_ptr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    for base in tl.range(tl.program_id(0) * BLOCK, N, NPROG * BLOCK):
        offs = base + cols
        mask = offs < N
        valid = mask
        src_off = tl.full((BLOCK,), 0, tl.int32)
        for d in tl.static_range(NDIM):
            osh = tl.load(out_shape_ptr + d)
            ostr = tl.load(out_stride_ptr + d)
            istr = tl.load(in_stride_ptr + d)
            vst = tl.load(valid_start_ptr + d)
            ven = tl.load(valid_end_ptr + d)
            coord = (offs // ostr) % osh
            valid = valid & (coord >= vst) & (coord < ven)
            src_off = src_off + (coord - vst) * istr
        val = tl.load(in_ptr + src_off, mask=valid, other=value)
        tl.store(out_ptr + offs, val, mask=mask)


# Row-batch pad kernel for rank <= 3 (the hot path).  Each work item is one
# (row, col-block) pair; the row base is a *scalar* decomposition over the
# leading dims, and every load/store stays a 1D affine ``base + arange``
# vector -- 2D strided tiles (``row[:, None] + col[None, :]``) degrade to
# per-lane gathers on GCU400 and are an order of magnitude slower.  Loads and
# stores are issued in UNROLL-sized batches (all loads of a batch are in
# flight before the first store consumes them) so the per-segment latency is
# amortized; this is the dominant term for large shapes (measured: (1024^3)
# fp16 56ms -> 3.8ms vs the old 2D-tile kernel).
@libentry()
@triton.jit(
    do_not_specialize=[
        "value",
        "n_rows",
        "out1",
        "n0",
        "n1",
        "in_last",
        "out_last",
        "pad_left",
        "vs0",
        "ve0",
        "vs1",
        "ve1",
        "n_col_blocks",
    ]
)
def constant_pad_nd_rows_kernel(
    in_ptr,
    out_ptr,
    value,
    n_rows,
    out1,
    n0,
    n1,
    in_last,
    out_last,
    pad_left,
    vs0,
    ve0,
    vs1,
    ve1,
    n_col_blocks,
    RANK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NCB1: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    total = n_rows * n_col_blocks
    cols = tl.arange(0, BLOCK_C)
    for base in tl.range(pid * UNROLL, total, GRID_SIZE * UNROLL):
        xs = ()
        for k in tl.static_range(UNROLL):
            chunk = base + k
            ok = chunk < total
            if RANK == 1:
                row = 0
                cb = chunk
                src_base = 0
                keep = (cb * BLOCK_C + cols >= pad_left) & (
                    cb * BLOCK_C + cols < pad_left + in_last
                )
            else:
                if NCB1:
                    row = chunk
                    cb = 0
                else:
                    row = chunk // n_col_blocks
                    cb = chunk - row * n_col_blocks
                if RANK == 3:
                    d0 = row // out1
                    d1 = row - d0 * out1
                    s0 = tl.minimum(tl.maximum(d0 - vs0, 0), n0 - 1)
                    s1 = tl.minimum(tl.maximum(d1 - vs1, 0), n1 - 1)
                    row_valid = (d0 >= vs0) & (d0 < ve0) & (d1 >= vs1) & (d1 < ve1)
                    src_base = (s0 * n1 + s1) * in_last
                else:
                    s0 = tl.minimum(tl.maximum(row - vs0, 0), n0 - 1)
                    row_valid = (row >= vs0) & (row < ve0)
                    src_base = s0 * in_last
                keep = (cols >= pad_left) & (cols < pad_left + in_last) & row_valid
            xs = xs + (
                tl.load(
                    in_ptr + src_base + (cb * BLOCK_C + cols - pad_left),
                    mask=keep & ok,
                    other=value,
                ),
            )
        for k in tl.static_range(UNROLL):
            chunk = base + k
            ok = chunk < total
            if RANK == 1:
                row = 0
                cb = chunk
            else:
                if NCB1:
                    row = chunk
                    cb = 0
                else:
                    row = chunk // n_col_blocks
                    cb = chunk - row * n_col_blocks
            tl.store(
                out_ptr + row * out_last + cb * BLOCK_C + cols,
                xs[k],
                mask=((cb * BLOCK_C + cols) < out_last) & ok,
            )


def _pad_rows_launch(src, out, in_shape, out_shape, valid_start, valid_end, value):
    ndim = len(in_shape)
    in_last = in_shape[-1]
    out_last = out_shape[-1]
    pad_left = valid_start[-1]
    if ndim == 1:
        n_rows = 1
        n0, n1, o1 = 1, 1, 1
        vs0, ve0, vs1, ve1 = 0, 1, 0, 1
        rank = 1
    elif ndim == 2:
        n_rows = out_shape[0]
        n0, n1, o1 = in_shape[0], 1, 1
        vs0, ve0, vs1, ve1 = valid_start[0], valid_end[0], 0, 1
        rank = 2
    else:
        n_rows = out_shape[0] * out_shape[1]
        n0, n1, o1 = in_shape[0], in_shape[1], out_shape[1]
        vs0, ve0, vs1, ve1 = (
            valid_start[0],
            valid_end[0],
            valid_start[1],
            valid_end[1],
        )
        rank = 3
    if rank == 1:
        block_c = 8192
        unroll = 4
    else:
        block_c = min(2048, triton.next_power_of_2(out_last))
        unroll = 8
    ncb = triton.cdiv(out_last, block_c)
    total = n_rows * ncb
    nprog = min(_GRID_SIZE, triton.cdiv(total, unroll))
    constant_pad_nd_rows_kernel[(nprog,)](
        src,
        out,
        value,
        n_rows,
        o1,
        n0,
        n1,
        in_last,
        out_last,
        pad_left,
        vs0,
        ve0,
        vs1,
        ve1,
        ncb,
        RANK=rank,
        BLOCK_C=block_c,
        GRID_SIZE=nprog,
        NCB1=(ncb == 1),
        UNROLL=unroll,
        num_warps=2,
    )


def constant_pad_nd(inp, pad_list, value=0):
    logger.debug("GEMS_ENFLAME CONSTANT_PAD_ND")
    ndim = inp.dim()
    pad_pairs = len(pad_list) // 2
    if pad_pairs == 0:
        return inp.clone()

    in_shape = inp.shape
    out_shape = list(in_shape)
    valid_start = [0] * ndim
    valid_end = [0] * ndim
    for i in range(ndim):
        valid_start[i] = 0
        valid_end[i] = in_shape[i]
    # aten pad order: [left_last, right_last, left_prev, right_prev, ...].
    # Negative pads crop; the same valid_start/valid_end formulas cover both.
    for i in range(pad_pairs):
        dim = ndim - 1 - i
        if dim < 0:
            break
        left = int(pad_list[2 * i])
        right = int(pad_list[2 * i + 1])
        out_shape[dim] = in_shape[dim] + left + right
        valid_start[dim] = left
        valid_end[dim] = left + in_shape[dim]

    out_numel = 1
    for d in out_shape:
        out_numel *= d
    if out_numel == 0:
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    src = inp.contiguous()

    if ndim >= 4:
        # High-rank path: persistent flat kernel with tensor metadata (the
        # scalar row decomposition below only covers rank <= 3).
        out_shape_t = torch.tensor(out_shape, dtype=torch.int32, device=inp.device)
        out_stride = list(out.stride())
        out_stride_t = torch.tensor(out_stride, dtype=torch.int32, device=inp.device)
        in_stride_t = torch.tensor(
            list(src.stride()), dtype=torch.int32, device=inp.device
        )
        vs_t = torch.tensor(valid_start, dtype=torch.int32, device=inp.device)
        ve_t = torch.tensor(valid_end, dtype=torch.int32, device=inp.device)
        block = 1024
        nprog = min(triton.cdiv(out_numel, block), _MAX_PROGRAMS)
        _pad_flat_kernel[(nprog,)](
            src,
            out,
            value,
            out_numel,
            out_shape_t,
            out_stride_t,
            in_stride_t,
            vs_t,
            ve_t,
            NDIM=ndim,
            BLOCK=block,
            NPROG=nprog,
            num_warps=4,
        )
        return out

    # Rank <= 3 hot path: scalar metadata, affine 1D loads/stores, batched
    # load-ahead.  The kernel indexes in int32; guard the address space (the
    # generic fallback handles anything at or beyond 2**31 elements/strides).
    in_numel = src.numel()
    meta_max = max(
        [out_numel, in_numel]
        + [abs(s) for s in src.stride()]
        + [out_shape[i] for i in range(ndim)]
        + [valid_end[i] for i in range(ndim)]
    )
    if meta_max >= 2**31:
        from flag_gems.ops.pad import pad as _generic_pad

        return _generic_pad(inp, pad_list, mode="constant", value=value)

    _pad_rows_launch(src, out, list(in_shape), out_shape, valid_start, valid_end, value)
    return out
