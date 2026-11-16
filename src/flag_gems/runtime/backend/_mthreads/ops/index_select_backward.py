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

from flag_gems.ops.index_select_backward import (
    index_select_backward as default_index_select_backward,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ---------------------------------------------------------------------------
# Contiguous fast path, dim == last (I == 1).
#
# out[o, j] = sum_{p : index[p] == j} grad[o, p]
#
# For a block of BO source rows and BD destination ids this is a (BO x BK) by
# (BK x BD) product with an exact 0/1 selection matrix, so the whole
# comparison workload becomes one tensor-core dot instead of n_out*n_index
# scalarized predicated gathers.  The grad block and the selection matrix are
# built from ordinary coalesced loads / compares, so no extra launch and no
# extra workspace are needed.
# ---------------------------------------------------------------------------
@triton.jit
def _isb_dot_last(
    grad_ptr,
    index_ptr,
    out_ptr,
    D: tl.constexpr,
    O: tl.constexpr,
    NI: tl.constexpr,
    BK: tl.constexpr,
    BO: tl.constexpr,
    BD: tl.constexpr,
    IS: tl.constexpr,
    FP32: tl.constexpr,
):
    jb = tl.program_id(0)
    ob = tl.program_id(1)
    j = jb * BD + tl.arange(0, BD)
    o = ob * BO + tl.arange(0, BO)
    k = tl.arange(0, BK)
    km = k < NI

    idx = tl.load(index_ptr + k * IS, mask=km, other=-1)
    g = tl.load(
        grad_ptr + (o[:, None] * NI + k[None, :]),
        mask=(o < O)[:, None] & km[None, :],
        other=0.0,
    )
    if FP32:
        s = (idx[:, None] == j[None, :]).to(tl.float32)
        acc = tl.dot(g, s, input_precision="ieee", out_dtype=tl.float32)
    else:
        s = (idx[:, None] == j[None, :]).to(grad_ptr.dtype.element_ty)
        acc = tl.dot(g, s, out_dtype=tl.float32)

    tl.store(
        out_ptr + (o[:, None] * D + j[None, :]),
        acc.to(out_ptr.dtype.element_ty),
        mask=(o < O)[:, None] & (j < D)[None, :],
    )


# ---------------------------------------------------------------------------
# Contiguous fast path, dim == last (I == 1), float32 only.
#
# The emulated fp32 tl.dot is expensive per dot instruction on this backend,
# so for a long index vector the predicated gather form is used instead: each
# program owns a block of destination lanes and walks the index in BP-sized
# tiles, masking the grad loads by the value comparison.
# ---------------------------------------------------------------------------
@triton.jit
def _isb_gather_last(
    grad_ptr,
    index_ptr,
    out_ptr,
    D: tl.constexpr,
    NI: tl.constexpr,
    IS: tl.constexpr,
    N_OUT: tl.constexpr,
    BLOCK: tl.constexpr,
    BP: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    om = o < N_OUT
    j = o % D
    base = (o // D) * NI

    acc = tl.zeros([BLOCK, BP], dtype=tl.float32)
    for p0 in tl.static_range(0, NI, BP):
        p = p0 + tl.arange(0, BP)
        pmask = p < NI
        idx = tl.load(index_ptr + p * IS, mask=pmask, other=-1).to(tl.int32)
        hit = (idx[None, :] == j[:, None]) & om[:, None]
        v = tl.load(grad_ptr + (base[:, None] + p[None, :]), mask=hit, other=0).to(
            tl.float32
        )
        acc += v

    res = tl.sum(acc, axis=1)
    tl.store(out_ptr + o, res.to(out_ptr.dtype.element_ty), mask=om)


# ---------------------------------------------------------------------------
# Contiguous fast path, dim not last (I > 1).
#
# out[(o, j, i)] = sum_{p : index[p] == j} grad[(o, p, i)]
#
# Same idea: a (BD x BK) selection block times a (BK x BI) grad block.  The
# grad block is coalesced along the inner axis and is read once per
# destination tile instead of once per (destination, p) pair.
# ---------------------------------------------------------------------------
@triton.jit
def _isb_dot_mid(
    grad_ptr,
    index_ptr,
    out_ptr,
    D: tl.constexpr,
    I: tl.constexpr,
    NI: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BI: tl.constexpr,
    IS: tl.constexpr,
    FP32: tl.constexpr,
):
    jb = tl.program_id(0)
    ib = tl.program_id(1)
    o = tl.program_id(2)
    j = jb * BD + tl.arange(0, BD)
    inner = ib * BI + tl.arange(0, BI)
    k = tl.arange(0, BK)
    km = k < NI
    im = inner < I

    idx = tl.load(index_ptr + k * IS, mask=km, other=-1)
    b = tl.load(
        grad_ptr + o * (NI * I) + (k[:, None] * I + inner[None, :]),
        mask=km[:, None] & im[None, :],
        other=0.0,
    )
    if FP32:
        a = (idx[None, :] == j[:, None]).to(tl.float32)
        acc = tl.dot(a, b, input_precision="ieee", out_dtype=tl.float32)
    else:
        a = (idx[None, :] == j[:, None]).to(grad_ptr.dtype.element_ty)
        acc = tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        out_ptr + o * (D * I) + (j[:, None] * I + inner[None, :]),
        acc.to(out_ptr.dtype.element_ty),
        mask=(j < D)[:, None] & im[None, :],
    )


# ---------------------------------------------------------------------------
# Fallback kernel: arbitrary grad strides, per-axis coordinate decode.
# ---------------------------------------------------------------------------
@triton.jit
def _isb_general_kernel(
    grad_ptr,
    index_ptr,
    out_ptr,
    n_out,
    n_index,
    istride,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    ip0,
    ip1,
    ip2,
    ip3,
    ip4,
    ip5,
    ip6,
    ip7,
    gs0,
    gs1,
    gs2,
    gs3,
    gs4,
    gs5,
    gs6,
    gs7,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    DIM: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < n_out

    base = tl.zeros([BLOCK], dtype=tl.int64)
    odv = tl.zeros([BLOCK], dtype=tl.int64)

    if R > 0:
        c = (o // ip0) % s0
        if DIM == 0:
            odv = c.to(tl.int64)
        else:
            base += c.to(tl.int64) * gs0
    if R > 1:
        c = (o // ip1) % s1
        if DIM == 1:
            odv = c.to(tl.int64)
        else:
            base += c.to(tl.int64) * gs1
    if R > 2:
        c = (o // ip2) % s2
        if DIM == 2:
            odv = c.to(tl.int64)
        else:
            base += c.to(tl.int64) * gs2
    if R > 3:
        c = (o // ip3) % s3
        if DIM == 3:
            odv = c.to(tl.int64)
        else:
            base += c.to(tl.int64) * gs3
    if R > 4:
        c = (o // ip4) % s4
        if DIM == 4:
            odv = c.to(tl.int64)
        else:
            base += c.to(tl.int64) * gs4
    if R > 5:
        c = (o // ip5) % s5
        if DIM == 5:
            odv = c.to(tl.int64)
        else:
            base += c.to(tl.int64) * gs5
    if R > 6:
        c = (o // ip6) % s6
        if DIM == 6:
            odv = c.to(tl.int64)
        else:
            base += c.to(tl.int64) * gs6
    if R > 7:
        c = (o // ip7) % s7
        if DIM == 7:
            odv = c.to(tl.int64)
        else:
            base += c.to(tl.int64) * gs7

    if DIM == 0:
        gs_dim = gs0
    elif DIM == 1:
        gs_dim = gs1
    elif DIM == 2:
        gs_dim = gs2
    elif DIM == 3:
        gs_dim = gs3
    elif DIM == 4:
        gs_dim = gs4
    elif DIM == 5:
        gs_dim = gs5
    elif DIM == 6:
        gs_dim = gs6
    else:
        gs_dim = gs7

    if MODE == 0:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
    elif MODE == 1:
        acc = tl.zeros([BLOCK], dtype=tl.float64)
    elif MODE == 2:
        acc = tl.zeros([BLOCK], dtype=tl.int32)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.int64)

    off = base
    p = 0
    while p < n_index:
        iv = tl.load(index_ptr + p * istride).to(tl.int64)
        hit = (iv == odv) & mask
        v = tl.load(grad_ptr + off, mask=hit, other=0)
        if MODE == 0:
            acc += v.to(tl.float32)
        elif MODE == 1:
            acc += v.to(tl.float64)
        else:
            acc += v
        off += gs_dim
        p += 1

    tl.store(out_ptr + o, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _mode_of(dtype):
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return 0
    if dtype == torch.float64:
        return 1
    if dtype == torch.int64:
        return 3
    return 2


_GBLOCK = 128
_PLANS = {}
_MISS = object()


def _p2(n):
    if n <= 16:
        return 16
    return 1 << (n - 1).bit_length()


def _build_dot_plan(sizes, d, dt, n_index, istride):
    """Return a cached launch plan, or None when the dot path cannot be used."""
    if dt not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    R = len(sizes)
    D = sizes[d]
    inner = 1
    for a in range(d + 1, R):
        inner *= sizes[a]
    outer = 1
    for a in range(d):
        outer *= sizes[a]
    if D <= 0 or inner <= 0 or outer <= 0:
        return None
    bk = _p2(n_index)
    if bk > 256:
        return None
    fp32 = dt == torch.float32
    if inner == 1:
        if fp32 and bk > 64:
            # The fp32 tl.dot is emulated on this backend and its cost grows
            # with the number of dot instructions and tiles, so a long index
            # vector is handled better by the predicated gather form.
            n_out = outer * D
            blk = 64
            bp = 32
            return (
                _isb_gather_last,
                ((n_out + blk - 1) // blk,),
                (D, n_index, istride, n_out, blk, bp),
                4,
                outer * n_index,
            )
        bo = 32 if outer >= 32 else 16
        bd = 64
        while bd > 16 and ((D + bd - 1) // bd) * ((outer + bo - 1) // bo) < 60:
            bd //= 2
        grid = ((D + bd - 1) // bd, (outer + bo - 1) // bo)
        return (
            _isb_dot_last,
            grid,
            (D, outer, n_index, bk, bo, bd, istride, fp32),
            4,
            outer * n_index,
        )
    # tl.arange needs a power-of-two row tile, so pick the BD whose
    # ceil(D/BD)*BD covers D with the least padding (ties prefer the larger
    # tile, i.e. fewer blocks and a wider dot).
    bd = 16
    pad_best = None
    for cand in (16, 32, 64):
        pad = ((D + cand - 1) // cand) * cand - D
        if pad_best is None or pad <= pad_best:
            pad_best = pad
            bd = cand
    bi = min(64, _p2(inner))
    nj = (D + bd - 1) // bd
    ni = (inner + bi - 1) // bi
    while bi > 32 and outer * nj * ni < 120:
        bi //= 2
        ni = (inner + bi - 1) // bi
    if bk * bd > 4096:
        bd = max(16, 4096 // bk)
    if bk * bi > 4096:
        bi = max(16, 4096 // bk)
    grid = ((D + bd - 1) // bd, (inner + bi - 1) // bi, outer)
    return (
        _isb_dot_mid,
        grid,
        (D, inner, n_index, bk, bd, bi, istride, fp32),
        4,
        outer * n_index * inner,
    )


def _run_general(grad, index, sizes, d, dt, n_index, istride):
    R = len(sizes)
    D = sizes[d]
    inner = 1
    for a in range(d + 1, R):
        inner *= sizes[a]
    outer = 1
    for a in range(d):
        outer *= sizes[a]
    n_out = outer * D * inner
    out = torch.empty(sizes, dtype=dt, device=grad.device)
    if n_out == 0 or n_index == 0:
        return out
    gd = grad.dim()
    gstride = grad.stride()
    sz = (list(sizes) + [1] * 8)[:8]
    ip = [1] * 8
    acc_ip = 1
    for a in range(7, -1, -1):
        ip[a] = acc_ip
        acc_ip *= sz[a]
    _isb_general_kernel[((n_out + _GBLOCK - 1) // _GBLOCK,)](
        grad,
        index,
        out,
        n_out,
        n_index,
        istride,
        sz[0],
        sz[1],
        sz[2],
        sz[3],
        sz[4],
        sz[5],
        sz[6],
        sz[7],
        ip[0],
        ip[1],
        ip[2],
        ip[3],
        ip[4],
        ip[5],
        ip[6],
        ip[7],
        int(gstride[0]) if gd > 0 else 0,
        int(gstride[1]) if gd > 1 else 0,
        int(gstride[2]) if gd > 2 else 0,
        int(gstride[3]) if gd > 3 else 0,
        int(gstride[4]) if gd > 4 else 0,
        int(gstride[5]) if gd > 5 else 0,
        int(gstride[6]) if gd > 6 else 0,
        int(gstride[7]) if gd > 7 else 0,
        BLOCK=_GBLOCK,
        R=min(R, 8),
        DIM=d,
        MODE=_mode_of(dt),
        num_warps=4,
    )
    return out


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def index_select_backward(grad, self_sizes, dim, index):
    logger.debug("GEMS_MTHREADS INDEX_SELECT_BACKWARD")

    if not _use_triton_kernel(grad):
        return default_index_select_backward(grad, self_sizes, dim, index)

    with torch_device_fn.device(grad.device):
        return _launch_index_select_backward(grad, self_sizes, dim, index)


def _launch_index_select_backward(grad, self_sizes, dim, index):
    if isinstance(self_sizes, int):
        sizes = (int(self_sizes),)
    else:
        sizes = tuple(int(s) for s in self_sizes)
    R = len(sizes)
    if R == 0:
        return torch.zeros((), dtype=grad.dtype, device=grad.device)
    d = int(dim)
    if d < 0:
        d += R
    if d < 0 or d >= R:
        raise IndexError(
            "Dimension out of range (expected to be in range of [%d, %d], but got %d)"
            % (-R, R - 1, int(dim))
        )
    dt = grad.dtype
    n_index = index.numel()
    istride = index.stride(0) if index.dim() > 0 else 1
    key = (sizes, d, dt, n_index, istride)
    plan = _PLANS.get(key, _MISS)
    if plan is _MISS:
        plan = _build_dot_plan(sizes, d, dt, n_index, istride)
        _PLANS[key] = plan
    if plan is not None:
        fn, grid, cargs, nw, expect = plan
        if grad.numel() == expect and grad.dim() == R and grad.is_contiguous():
            out = torch.empty(sizes, dtype=dt, device=grad.device)
            fn[grid](grad, index, out, *cargs, num_warps=nw)
            return out
    return _run_general(grad, index, sizes, d, dt, n_index, istride)
