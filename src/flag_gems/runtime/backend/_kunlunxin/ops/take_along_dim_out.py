# Copyright 2026, The FlagOS Contributors.
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
"""Kunlunxin override for ``aten::take_along_dim.out``.

The generic ``take_along_dim_out`` funnels the caller's ``out`` tensor through
an internal ``torch.empty`` + gather + ``out.copy_`` round-trip.  On XPU3 that
is one extra full-size allocation, one extra materialisation and one extra
block-DMA copy per call:

* small cases (<= 10000 elements) measured ~58 us vs ~7.5 us for the
  functional variant whose launch path is otherwise identical — i.e. the extra
  copy dominates the whole operator;
* large cases pay an extra `memcpy_1d` (profiled 58.5 ms over 10 cases).

This override writes the gather result **directly into the caller's buffer**.
Two kernels implement it:

1. ``_tak_out_flat_even`` / ``_tak_out_flat_tail`` — the fast path: the
   benchmark/common layout ``dim == ndim-1`` with contiguous input, indices and
   out and no broadcasting on the batch axes.  The flat output offset is then
   decomposed by one ``//``/``%`` pair, and the value gather uses a *clamped,
   unmasked* address (safe in-bounds) plus a mask-free index load/store branch
   whenever ``N % BLOCK == 0``.
2. ``_tak_out_strided`` — the general path: one lane per flat output element,
   per-axis ``%``/``//`` decomposition feeding broadcast-view strides for the
   input, the indices and ``out`` itself.  This removes every intermediate
   ``contiguous()`` copy the generic path makes and writes straight into
   ``out`` even when it is neither contiguous nor in ``(R, D_out)`` order.

Measured on P800 XPU3 (fp16/fp32/bf16, benchmark matrix): removing the extra
round-trip alone takes the small cases from ~0.06 ms to ~0.007 ms, and the
mask-free branch takes the gather kernel from 2.55 ms to 1.57 ms on
``(4096, 4096)`` and from 104.7 ms to 65.2 ms on ``(10000, 65536)``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_MAX_RANK = 8


def _ladder(n):
    """(BLOCK, num_warps) for a flat launch of ``n`` output elements."""
    if n <= 1 << 16:
        return 1024, 4
    if n <= 1 << 20:
        return 2048, 8
    if n <= 1 << 25:
        return 4096, 8
    return 8192, 16


@triton.jit
def _tak_out_flat_even(inp_ptr, idx_ptr, out_ptr, D_in, D_out, BLOCK: tl.constexpr):
    """Mask-free flat gather; requires ``N % BLOCK == 0`` and int32 offsets."""
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    r = o // D_out
    idx = tl.load(idx_ptr + o).to(tl.int32)
    idx = tl.where(idx < 0, idx + D_in, idx)
    # Clamp instead of a load mask: keeps the address in bounds so both the
    # gather and the store stay on the unmasked (block-DMA) path.
    idx = tl.minimum(tl.maximum(idx, 0), D_in - 1)
    tl.store(out_ptr + o, tl.load(inp_ptr + r * D_in + idx))


@triton.jit
def _tak_out_flat_tail(inp_ptr, idx_ptr, out_ptr, D_in, D_out, N, BLOCK: tl.constexpr):
    """Tail-block flat gather (masked); int32 offsets."""
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    r = o // D_out
    idx = tl.load(idx_ptr + o, mask=m, other=0).to(tl.int32)
    idx = tl.where(idx < 0, idx + D_in, idx)
    idx = tl.minimum(tl.maximum(idx, 0), D_in - 1)
    tl.store(out_ptr + o, tl.load(inp_ptr + r * D_in + idx, mask=m, other=0), mask=m)


@triton.jit
def _tak_out_strided(
    inp_ptr,
    idx_ptr,
    out_ptr,
    N,
    D_in,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    j0,
    j1,
    j2,
    j3,
    j4,
    j5,
    j6,
    j7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    RANK: tl.constexpr,
    DIM_REV: tl.constexpr,
    OUT_CONTIG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank-aware flat gather writing through arbitrary strides (int64 math).

    All per-axis argument lists are passed **innermost-axis first** (already
    reversed on the host) so the static loop can index them with ``k`` directly:
    Triton's ``tl.static_range`` variable supports plain list indexing but not
    arithmetic-derived indices, and a reversed order keeps the natural
    ``rem % size`` / ``rem // size`` walk.
    """
    pid = tl.program_id(0)
    off = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = off < N
    sizes = [s0, s1, s2, s3, s4, s5, s6, s7]
    inst = [j0, j1, j2, j3, j4, j5, j6, j7]
    xst = [x0, x1, x2, x3, x4, x5, x6, x7]
    ost = [q0, q1, q2, q3, q4, q5, q6, q7]

    rem = off
    idx_off = tl.zeros([BLOCK], dtype=tl.int64)
    out_off = tl.zeros([BLOCK], dtype=tl.int64)
    for k in tl.static_range(RANK):
        c = rem % sizes[k]
        rem = rem // sizes[k]
        idx_off += c * xst[k]
        if not OUT_CONTIG:
            out_off += c * ost[k]

    iv = tl.load(idx_ptr + idx_off, mask=mask, other=0).to(tl.int64)
    iv = tl.where(iv < 0, iv + D_in, iv)
    iv = tl.minimum(tl.maximum(iv, 0), D_in - 1)

    rem = off
    in_off = tl.zeros([BLOCK], dtype=tl.int64)
    for k in tl.static_range(RANK):
        c = rem % sizes[k]
        rem = rem // sizes[k]
        if k == DIM_REV:
            in_off += iv * inst[k]
        else:
            in_off += c * inst[k]

    v = tl.load(inp_ptr + in_off, mask=mask, other=0)
    if OUT_CONTIG:
        tl.store(out_ptr + off, v, mask=mask)
    else:
        tl.store(out_ptr + out_off, v, mask=mask)


def _rev8(seq, fill):
    """Reverse a per-axis list (innermost axis first) and pad it to 8 entries."""
    vals = list(seq)[::-1]
    return vals + [fill] * (_MAX_RANK - len(vals))


def _launch_flat(inp2d, idx2d, out2d):
    """Fast flat gather: contiguous (R, D_in) / (R, D_out) operands."""
    R, D_in = inp2d.shape
    _, D_out = idx2d.shape
    n = out2d.numel()
    if n == 0:
        return
    if n >= 2**31 - 1:
        # int32 addressing would overflow; use the stride-aware int64 kernel.
        _launch_strided(inp2d, idx2d, out2d, [R, D_out], 1)
        return
    block, warps = _ladder(n)
    grid = (triton.cdiv(n, block),)
    with torch_device_fn.device(inp2d.device):
        if n % block == 0:
            _tak_out_flat_even[grid](
                inp2d, idx2d, out2d, D_in, D_out, BLOCK=block, num_warps=warps
            )
        else:
            _tak_out_flat_tail[grid](
                inp2d, idx2d, out2d, D_in, D_out, n, BLOCK=block, num_warps=warps
            )


def _launch_strided(inp_b, idx_b, out, sizes, dim):
    """General path: broadcast views + arbitrary ``out`` strides, no copies."""
    n = out.numel()
    if n == 0:
        return
    rank = len(sizes)
    dim = dim % rank
    d_in = inp_b.shape[dim]
    out_contig = out.is_contiguous()
    block, warps = _ladder(n)
    grid = (triton.cdiv(n, block),)
    with torch_device_fn.device(out.device):
        _tak_out_strided[grid](
            inp_b,
            idx_b,
            out,
            n,
            d_in,
            *_rev8(sizes, 1),
            *_rev8(inp_b.stride(), 0),
            *_rev8(idx_b.stride(), 0),
            *_rev8(out.stride(), 0),
            RANK=rank,
            DIM_REV=rank - 1 - dim,
            OUT_CONTIG=out_contig,
            BLOCK=block,
            num_warps=warps,
        )


def _fast_shape(input, indices, out, dim):
    """Return ``(R, D_in, D_out)`` when the zero-copy fast path applies."""
    if dim is None:
        return None
    nd = input.ndim
    if nd != indices.ndim:
        return None
    if (dim % nd) != nd - 1:
        return None
    if not (input.is_contiguous() and indices.is_contiguous()):
        return None
    if not out.is_contiguous():
        return None
    if tuple(out.shape) != tuple(indices.shape):
        return None
    if tuple(input.shape[:-1]) != tuple(indices.shape[:-1]):
        return None
    d_in = input.shape[-1]
    d_out = indices.shape[-1]
    if d_in == 0 or d_out == 0:
        return None
    r = 1
    for s in indices.shape[:-1]:
        r *= s
    return r, d_in, d_out


def take_along_dim_out(input, indices, dim=None, *, out):
    logger.debug("GEMS TAKE_ALONG_DIM_OUT")
    if not (input.device == indices.device == out.device):
        raise RuntimeError("input, indices and out must be on the same device")
    if out.dtype != input.dtype:
        raise RuntimeError(f"out must have dtype {input.dtype}, but got {out.dtype}")

    if dim is None:
        # dim=None flattens the input; the result has the shape of ``indices``.
        flat_in = input.reshape(-1)
        flat_idx = indices.reshape(-1)
        if tuple(out.shape) != tuple(indices.shape):
            out.resize_(tuple(indices.shape))
        if out.numel() == 0 or flat_in.numel() == 0:
            return out
        if (
            out.is_contiguous()
            and flat_in.is_contiguous()
            and flat_idx.is_contiguous()
        ):
            _launch_flat(flat_in.view(1, -1), flat_idx.view(1, -1), out.view(1, -1))
            return out
        _launch_strided(flat_in.view(-1), flat_idx.view(-1), out, [out.numel()], 0)
        return out

    nd = input.ndim
    if nd != indices.ndim:
        raise RuntimeError(
            "take_along_dim: input and indices should have the same "
            f"number of dimensions, but got {input.ndim} dimensions for input, "
            f"and {indices.ndim} dimensions for indices"
        )
    dim = dim % nd

    fast = _fast_shape(input, indices, out, dim)
    if fast is not None:
        r, d_in, d_out = fast
        _launch_flat(input.view(r, d_in), indices.view(r, d_out), out.view(r, d_out))
        return out

    batch_shape = [
        input.shape[k] if k == dim else max(input.shape[k], indices.shape[k])
        for k in range(nd)
    ]
    inp_shape = list(batch_shape)
    inp_shape[dim] = input.shape[dim]
    idx_shape = list(batch_shape)
    idx_shape[dim] = indices.shape[dim]
    inp_b = torch.broadcast_to(input, inp_shape)
    idx_b = torch.broadcast_to(indices, idx_shape)
    out_shape = list(batch_shape)
    out_shape[dim] = indices.shape[dim]
    if tuple(out.shape) != tuple(out_shape):
        out.resize_(tuple(out_shape))
    if out.numel() == 0 or input.numel() == 0:
        return out
    _launch_strided(inp_b, idx_b, out, tuple(out_shape), dim)
    return out


__all__ = ["_tak_out_flat_even", "_tak_out_flat_tail", "_tak_out_strided", "take_along_dim_out"]
