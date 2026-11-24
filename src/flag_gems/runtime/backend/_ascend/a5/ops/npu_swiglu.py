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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _swiglu_last(
    x_ptr,
    out_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    NCOL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // NCOL
    col = (pid % NCOL) * BLOCK_H + tl.arange(0, BLOCK_H)
    in_off = row * D + col
    out_off = row * H + col
    if EXACT:
        g = tl.load(x_ptr + in_off).to(tl.float32)
        gate = tl.load(x_ptr + in_off + H).to(tl.float32)
        y = (g * tl.sigmoid(g)) * gate
        tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty))
    else:
        m = col < H
        g = tl.load(x_ptr + in_off, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + in_off + H, mask=m, other=0.0).to(tl.float32)
        y = (g * tl.sigmoid(g)) * gate
        tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=m)


# Large shapes: one [ROWS, BLOCK_H] strided block per operand, which lowers to a
# single 2-D GM->UB DMA per operand (load_gm_to_ubuf_2d_bfloat16_t) instead of
# ROWS narrow copies.  The configuration is pinned, not searched: the in-process
# autotune pick was not reproducible between Eval runs, and ROWS=8/BLOCK_H=512
# (zero masked lanes, H=1536 = 3 exact tiles) is the variant that reproduces the
# fast result.  num_stages stays 1 because the captured IR shows no prefetch.
_LARGE_ROWS = 32
_LARGE_BLOCK = 512


@triton.jit
def _swiglu_rows(
    x_ptr,
    out_ptr,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS: tl.constexpr,
):
    NCOL: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    pid = tl.program_id(0)
    row = (pid // NCOL) * ROWS + tl.arange(0, ROWS)
    col = (pid % NCOL) * BLOCK_H + tl.arange(0, BLOCK_H)
    in_off = row[:, None] * D + col[None, :]
    out_off = row[:, None] * H + col[None, :]
    if R % ROWS == 0:
        if H % BLOCK_H == 0:
            g = tl.load(x_ptr + in_off).to(tl.float32)
            gate = tl.load(x_ptr + in_off + H).to(tl.float32)
            y = (g * tl.sigmoid(g)) * gate
            tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty))
        else:
            m = col[None, :] < H
            g = tl.load(x_ptr + in_off, mask=m, other=0.0).to(tl.float32)
            gate = tl.load(x_ptr + in_off + H, mask=m, other=0.0).to(tl.float32)
            y = (g * tl.sigmoid(g)) * gate
            tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=m)
    else:
        if H % BLOCK_H == 0:
            m = row[:, None] < R
        else:
            m = (row[:, None] < R) & (col[None, :] < H)
        g = tl.load(x_ptr + in_off, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + in_off + H, mask=m, other=0.0).to(tl.float32)
        y = (g * tl.sigmoid(g)) * gate
        tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _swiglu_rows_pp(
    x_ptr,
    out_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS: tl.constexpr,
    SUB: tl.constexpr,
    NSUB: tl.constexpr,
):
    """Same footprint as _swiglu_rows (ROWS x BLOCK_H per program, exact tiles)
    but the program body is a pipelined loop over NSUB row sub-tiles so that the
    next sub-tile's GM->UB descriptors can issue while the current one is in the
    vector pipe."""
    NCOL: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    pid = tl.program_id(0)
    row0 = (pid // NCOL) * ROWS
    col = (pid % NCOL) * BLOCK_H + tl.arange(0, BLOCK_H)
    for t in tl.range(0, NSUB, num_stages=2):
        row = row0 + t * SUB + tl.arange(0, SUB)
        in_off = row[:, None] * D + col[None, :]
        out_off = row[:, None] * H + col[None, :]
        g = tl.load(x_ptr + in_off).to(tl.float32)
        gate = tl.load(x_ptr + in_off + H).to(tl.float32)
        y = (g * tl.sigmoid(g)) * gate
        tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _swiglu_generic(
    x_ptr,
    out_ptr,
    so,
    sc,
    sk,
    oo,
    oc,
    inner,
    loff,
    hi,
    total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < total
    o = i // hi
    rem = i % hi
    j = rem // inner
    k = rem % inner
    base = o * so + j * sc + k * sk
    g = tl.load(x_ptr + base, mask=m, other=0.0).to(tl.float32)
    gate = tl.load(x_ptr + base + loff, mask=m, other=0.0).to(tl.float32)
    y = (g * tl.sigmoid(g)) * gate
    tl.store(out_ptr + o * oo + j * oc + k, y.to(out_ptr.dtype.element_ty), mask=m)


# Shapes at or above this output size use the pinned multi-row descriptor; below
# it the single-tile kernel is kept, because both consolidating and fragmenting
# those shapes was measured to be slower.
_LARGE = 1 << 17
# Mid shapes (several column tiles, far below _LARGE, e.g. 8x3072) group
# _MID_ROWS rows per program so each program issues one [2,512] 2-D descriptor
# per operand: grid cdiv(rows,2)*ncol = 12 instead of rows*ncol = 24, with
# num_warps 4 and num_stages 1.  Single-column-tile shapes (H<=512) keep the
# single-tile kernel unchanged.
_MID_ROWS = 8
# Warp count for the mid-class launch; 4 vs 2 measured neutral.
_MID_WARPS = 4
# Column-tile width for the mid class.  _MID_ROWS x _MID_BLOCK is held at 1024
# elements per program and the grid at 12, so this constant only changes the
# tile aspect: [8,128] here, [4,256] in round 8 (the kept best), [2,512] in
# round 3.  The branch is gated on the default column tiling (ncol > 1) so
# single-column-tile shapes such as 8x512 keep their unchanged _swiglu_last
# launch no matter how narrow _MID_BLOCK becomes.
_MID_BLOCK = 128


def _block_h(h):
    if h > 512:
        return 512
    return 1 << max(0, (h - 1).bit_length())


def _large_block_h(h):
    return min(_LARGE_BLOCK, 1 << max(0, (h - 1).bit_length()))


def _launch(input, dim=-1):
    x = input
    nd = x.dim()
    d = dim + nd if dim < 0 else dim
    H = x.shape[d] // 2
    shape = list(x.shape)
    shape[d] = H
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    if H == 0 or x.numel() == 0:
        return out

    if d == nd - 1 and x.is_contiguous():
        rows = x.numel() // (2 * H)
        if rows * H >= _LARGE:
            bh = _large_block_h(H)
            ncol = (H + bh - 1) // bh
            if rows % _LARGE_ROWS == 0 and H % bh == 0:
                _swiglu_rows_pp[(triton.cdiv(rows, _LARGE_ROWS) * ncol,)](
                    x,
                    out,
                    H,
                    2 * H,
                    bh,
                    _LARGE_ROWS,
                    _LARGE_ROWS // 2,
                    2,
                    num_warps=8,
                    num_stages=2,
                )
            else:
                _swiglu_rows[(triton.cdiv(rows, _LARGE_ROWS) * ncol,)](
                    x,
                    out,
                    rows,
                    H,
                    2 * H,
                    bh,
                    _LARGE_ROWS,
                    num_warps=8,
                    num_stages=1,
                )
        else:
            bh = _block_h(H)
            ncol = (H + bh - 1) // bh
            mid_ncol = (H + _MID_BLOCK - 1) // _MID_BLOCK
            if ncol > 1 and rows % _MID_ROWS == 0:
                # Shapes that need several column tiles but are far below the
                # large threshold (the 8x3072 class) group _MID_ROWS rows into
                # one program: 2-D descriptors, grid cdiv(rows,4)*6 = 12 with
                # 1024 elements per program, num_warps 4, num_stages 1.
                _swiglu_rows[(triton.cdiv(rows, _MID_ROWS) * mid_ncol,)](
                    x,
                    out,
                    rows,
                    H,
                    2 * H,
                    _MID_BLOCK,
                    _MID_ROWS,
                    num_warps=_MID_WARPS,
                    num_stages=1,
                )
            else:
                _swiglu_last[(rows * ncol,)](
                    x,
                    out,
                    H=H,
                    D=2 * H,
                    NCOL=ncol,
                    BLOCK_H=bh,
                    EXACT=ncol * bh == H,
                    num_warps=4,
                )
    else:
        inner = 1
        for s in x.shape[d + 1 :]:
            inner *= s
        total = out.numel()
        _swiglu_generic[(triton.cdiv(total, 1024),)](
            x,
            out,
            x.stride(0) if nd > 0 else 1,
            x.stride(d),
            x.stride(nd - 1) if d < nd - 1 else 1,
            out.stride(0) if nd > 0 else 1,
            out.stride(d),
            inner,
            H * x.stride(d),
            H * inner,
            total,
            BLOCK=1024,
        )
    return out


def npu_swiglu(input, dim=-1):
    """Single-input SwiGLU along the last dimension of a BF16 matrix."""
    if (
        input.ndim != 2
        or dim not in (-1, 1)
        or input.shape[-1] % 2
        or not input.numel()
    ):
        raise ValueError("expected a nonempty matrix with an even last dimension")
    if input.dtype != torch.bfloat16 or not input.is_contiguous():
        raise ValueError("expected a contiguous BF16 input")
    if input.device.type != "npu":
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(input.device):
        return _launch(input, dim)
