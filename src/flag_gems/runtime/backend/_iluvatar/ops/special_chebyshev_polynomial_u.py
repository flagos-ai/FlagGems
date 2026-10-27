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


@triton.jit
def _bcast_offsets(
    offs64,
    OUT_SHAPE: tl.constexpr,
    X_SHAPE: tl.constexpr,
    X_STR: tl.constexpr,
    N_SHAPE: tl.constexpr,
    N_STR: tl.constexpr,
    RANK: tl.constexpr,
):
    x_off = tl.zeros(offs64.shape, dtype=tl.int64)
    n_off = tl.zeros(offs64.shape, dtype=tl.int64)
    rem = offs64
    for d in tl.static_range(RANK):
        dim = OUT_SHAPE[RANK - 1 - d]
        idx = rem % dim
        rem = rem // dim
        x_off += (idx % X_SHAPE[RANK - 1 - d]) * X_STR[RANK - 1 - d]
        n_off += (idx % N_SHAPE[RANK - 1 - d]) * N_STR[RANK - 1 - d]
    return x_off, n_off


# ---------------- scalar-n fast path (n is a python int in [0, 5]) ----------------
@triton.jit
def _scalar_kernel(
    x_ptr,
    out_ptr,
    numel,
    N_VAL: tl.constexpr,
    X_DIRECT: tl.constexpr,
    NO_MASK: tl.constexpr,
    OUT_SHAPE: tl.constexpr,
    X_SHAPE: tl.constexpr,
    X_STR: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if X_DIRECT:
        if NO_MASK:
            offs64 = offs.to(tl.int64)
            xv = tl.load(x_ptr + offs64)
        else:
            offs64 = offs.to(tl.int64)
            mask = offs64 < numel
            xv = tl.load(x_ptr + offs64, mask=mask, other=0.0)
    else:
        offs64 = offs.to(tl.int64)
        mask = offs64 < numel
        x_off, _ = _bcast_offsets(
            offs64, OUT_SHAPE, X_SHAPE, X_STR, (1,) * RANK, (0,) * RANK, RANK
        )
        xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    ax = tl.abs(xv)
    if N_VAL == 0:
        res = tl.full([BLOCK], 1.0, dtype=xv.dtype)
    elif N_VAL == 1:
        res = xv + xv
    else:
        p = tl.full([BLOCK], 1.0, dtype=xv.dtype)  # U_0
        q = xv + xv  # U_1
        for k in tl.static_range(2, N_VAL + 1):
            r = (xv + xv) * q - p
            p = q
            q = r
        res = q
    # |x| == 1: U_n(1) = n+1, U_n(-1) = (-1)^n (n+1)
    one = tl.full([BLOCK], 1.0, dtype=xv.dtype)
    val = (N_VAL + 1) * one
    val_abs1 = tl.where((xv > 0.0) | (N_VAL % 2 == 0), val, -val)
    res = tl.where(ax == 1.0, val_abs1, res)
    if X_DIRECT:
        if NO_MASK:
            tl.store(out_ptr + offs64, res)
        else:
            tl.store(out_ptr + offs64, res, mask=mask)
    else:
        tl.store(out_ptr + offs64, res, mask=mask)


# ---------------- tensor-n path ----------------
@triton.jit
def _reduce_kernel(
    n_ptr,
    pmax_ptr,
    pbad_ptr,
    numel,
    N_SHAPE: tl.constexpr,
    N_STR: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    n_off = tl.zeros([BLOCK], dtype=tl.int64)
    rem = offs.to(tl.int64)
    for d in tl.static_range(RANK):
        idx = rem % N_SHAPE[RANK - 1 - d]
        rem = rem // N_SHAPE[RANK - 1 - d]
        n_off += idx * N_STR[RANK - 1 - d]
    nv = tl.load(n_ptr + n_off, mask=mask, other=0)
    n_i = nv.to(tl.int64)
    bmax = tl.max(n_i, axis=0)
    bad = (n_i < 0) | (n_i > 5)
    bbad = tl.max(bad.to(tl.int64), axis=0)
    tl.store(pmax_ptr + pid, bmax)
    tl.store(pbad_ptr + pid, bbad)


@triton.jit
def _finalize_kernel(
    pmax_ptr, pbad_ptr, max_ptr, bad_ptr, nblocks, BLOCK: tl.constexpr
):
    acc_m = tl.zeros([BLOCK], dtype=tl.int64)
    acc_b = tl.zeros([BLOCK], dtype=tl.int64)
    nchunks = (nblocks + BLOCK - 1) // BLOCK
    for c in range(0, nchunks):
        offs = c * BLOCK + tl.arange(0, BLOCK)
        msk = offs < nblocks
        acc_m = tl.maximum(acc_m, tl.load(pmax_ptr + offs, mask=msk, other=0))
        acc_b = tl.maximum(acc_b, tl.load(pbad_ptr + offs, mask=msk, other=0))
    tl.store(max_ptr, tl.max(acc_m, axis=0))
    tl.store(bad_ptr, tl.max(acc_b, axis=0))


@triton.jit
def _main_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    max_ptr,
    numel,
    OUT_SHAPE: tl.constexpr,
    X_SHAPE: tl.constexpr,
    X_STR: tl.constexpr,
    N_SHAPE: tl.constexpr,
    N_STR: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x_off, n_off = _bcast_offsets(
        offs.to(tl.int64), OUT_SHAPE, X_SHAPE, X_STR, N_SHAPE, N_STR, RANK
    )
    xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    nv = tl.load(n_ptr + n_off, mask=mask, other=0)
    n_i = nv.to(tl.int64)
    max_n = tl.load(max_ptr).to(tl.int32)
    ax = tl.abs(xv)
    res = tl.zeros([BLOCK], dtype=xv.dtype)
    res = tl.where(n_i == 0, 1.0, res)
    res = tl.where(n_i == 1, xv + xv, res)
    val_abs1 = tl.where(
        (xv > 0.0) | (n_i % 2 == 0), (n_i + 1).to(xv.dtype), -((n_i + 1).to(xv.dtype))
    )
    res = tl.where(ax == 1.0, val_abs1, res)
    needs_rec = (n_i >= 2) & (ax != 1.0)
    p = tl.full([BLOCK], 1.0, dtype=xv.dtype)
    q = xv + xv
    for k in range(2, max_n + 1):
        r = (xv + xv) * q - p
        res = tl.where((k == n_i) & needs_rec, r, res)
        p = q
        q = r
    res = tl.where(n_i < 0, 0.0, res)
    tl.store(out_ptr + offs, res, mask=mask)


def _pad(shape, strides, rank):
    p = rank - len(shape)
    return (1,) * p + tuple(shape), (0,) * p + tuple(strides)


def _raise_bad_n():
    raise ValueError("n must be in [0, 5]")


def _scalar_run(x, nv):
    out_shape = tuple(x.shape)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    numel = out.numel()
    if numel == 0:
        return out
    BLOCK = 1024
    x_direct = x.is_contiguous()
    if x_direct:
        # contiguous scalar-n path: 1D linear indexing, no stride metadata needed
        rank = 1
        no_mask = numel % BLOCK == 0
        x_shape = (numel,)
        x_str = (1,)
    else:
        rank = x.dim()
        no_mask = False
        x_shape, x_str = _pad(x.shape, x.stride(), rank)
    _scalar_kernel[(triton.cdiv(numel, BLOCK),)](
        x,
        out,
        numel,
        N_VAL=nv,
        X_DIRECT=x_direct,
        NO_MASK=no_mask,
        OUT_SHAPE=out_shape,
        X_SHAPE=x_shape,
        X_STR=x_str,
        RANK=rank,
        BLOCK=BLOCK,
        num_warps=16,
    )
    return out


def special_chebyshev_polynomial_u(x, n):
    # scalar n (python int, numpy scalar, or 0-dim/1-elem tensor): fast path
    if isinstance(n, int):
        if n < 0 or n > 5:
            _raise_bad_n()
        return _scalar_run(x, n)

    if not isinstance(n, torch.Tensor):
        n = torch.as_tensor(n, dtype=torch.int64, device=x.device)

    if n.numel() == 1:
        nv = int(n.item())
        if nv < 0 or nv > 5:
            _raise_bad_n()
        return _scalar_run(x, nv)

    # tensor-n path: reduce to get max n and out-of-range flag
    out_shape = torch.broadcast_shapes(x.shape, n.shape)
    rank = max(x.dim(), n.dim(), len(out_shape))
    x_shape, x_str = _pad(x.shape, x.stride(), rank)
    n_shape, n_str = _pad(n.shape, n.stride(), rank)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    numel = out.numel()
    if numel == 0:
        return out
    BLOCK = 1024
    nblocks = triton.cdiv(numel, BLOCK)
    pmax = torch.empty((nblocks,), dtype=torch.int64, device=x.device)
    pbad = torch.empty((nblocks,), dtype=torch.int64, device=x.device)
    max_buf = torch.empty((1,), dtype=torch.int64, device=x.device)
    bad_buf = torch.empty((1,), dtype=torch.int64, device=x.device)
    _reduce_kernel[(nblocks,)](
        n, pmax, pbad, numel, N_SHAPE=n_shape, N_STR=n_str, RANK=rank, BLOCK=BLOCK
    )
    _finalize_kernel[(1,)](pmax, pbad, max_buf, bad_buf, nblocks, BLOCK=1024)
    if bad_buf.item() != 0:
        _raise_bad_n()
    _main_kernel[(triton.cdiv(numel, BLOCK),)](
        x,
        n,
        out,
        max_buf,
        numel,
        OUT_SHAPE=out_shape,
        X_SHAPE=x_shape,
        X_STR=x_str,
        N_SHAPE=n_shape,
        N_STR=n_str,
        RANK=rank,
        BLOCK=BLOCK,
    )
    return out
