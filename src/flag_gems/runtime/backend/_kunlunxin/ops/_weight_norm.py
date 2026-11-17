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

"""Kunlunxin XPU3 override for ``aten::_weight_norm``.

Semantics (deprecated weight-norm, no eps): for every coordinate ``j`` along
``dim``, ``norm[j] = sqrt(sum over all other axes of v**2)`` and
``out = v * (g[j] / norm[j])``.

Why this override exists
------------------------
The previous backend routed ``_weight_norm`` through
``weight_norm_interface`` whose ``dim == last`` kernel used a serial
``BLOCK_ROW_SIZE = 1`` loop (68 % of the measured profile; ~40 ms/call and
0.005x speedup on wide columns), and whose ``dim == 0`` path for ``N > 8192``
fell back to plain torch primitives (a red-line fallback).  On XPU3 the
following facts constrain a rewrite (measured; see the wiki):

* ``tl.sum(axis=0)`` on a 2-D tile is rejected / lowers to ``uni_sram`` OOB;
* ``tl.sum(axis=1)`` is silently wrong unless the reduced width is a multiple
  of 8;
* a ``tl.dot`` against an all-ones matrix is the fast, correct way to reduce
  along the otherwise-banned axis;
* an unmasked 2-D load with clamped indices breaks the dot pipeline
  (``Unsupported dma src data type: i1``), while a *masked* load with clamped
  indices is fine -- so every load clamps its address and also masks it, so the
  masked load never reads past the tensor.

Strategy
--------
``dim == 0``  (reduce the contiguous tail):
  * ``N == 1``            -> dedicated flat kernel (``out = v * g / |v|``).
  * ``N <= 8192``         -> fused multi-row ``[BS, BN]`` tile reduce
    (``BN`` padded to ``max(8, next_pow2(N))`` so the axis-1 width is a multiple
    of 8; ``BS`` is the largest power-of-two divisor of ``M`` to keep the fast
    unmasked-free path).
  * ``N > 8192``          -> ``tl.dot`` row reduction + one-row-per-program
    apply (no torch fallback).
``dim == last``  (reduce the strided rows): ``tl.dot(ones, x*x)`` column
reduction, split over row chunks when ``M > 8192`` so wide/short and
tall/narrow both stay parallel; scale is folded into the flat ``off % K``
apply.

Only ``src/.../_kunlunxin/`` is touched.  The backward reuses the existing
``weight_norm_interface_backward`` / ``weight_norm_except_dim_backward`` Triton
kernels; the forward saves the per-group norm exactly as they expect.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .weight_norm import weight_norm_except_dim, weight_norm_except_dim_backward
from .weightnorm import weight_norm_interface_backward

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _npo2(x):
    return max(1, triton.next_power_of_2(int(x)))


# --------------------------------------------------------------------------- #
# dim == 0
# --------------------------------------------------------------------------- #
@triton.jit
def _wn_d0_scale1_kernel(v, g, out, M, GS: tl.constexpr, BLOCK: tl.constexpr):
    """N == 1: out[i] = v[i] * g[i] / |v[i]|."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < M
    x = tl.load(v + i, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(g + i * GS, mask=mask, other=0.0).to(tl.float32)
    sc = gv / tl.abs(x)
    tl.store(out + i, (x * sc).to(out.dtype.element_ty), mask=mask)


@triton.jit
def _wn_d0_row_kernel(
    v,
    g,
    out,
    nrm,
    M,
    N: tl.constexpr,
    BN: tl.constexpr,
    GS: tl.constexpr,
    STORE_N: tl.constexpr,
):
    """One row per program with a 1-D tile (avoids the 1-row 2-D-tile bug)."""
    r = tl.program_id(0)
    n = tl.arange(0, BN)
    mask = n < N
    x = tl.load(v + r * N + n, mask=mask, other=0.0).to(tl.float32)
    s = tl.sqrt(tl.sum(x * x, axis=0))
    gv = tl.load(g + r * GS).to(tl.float32)
    if STORE_N:
        tl.store(nrm + r, s)
    tl.store(out + r * N + n, (x * (gv / s)).to(out.dtype.element_ty), mask=mask)


@triton.jit
def _wn_d0_multi_kernel(
    v,
    g,
    out,
    nrm,
    M,
    N: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    GS: tl.constexpr,
    STORE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BS + tl.arange(0, BS)
    n = tl.arange(0, BN)
    off = rows[:, None] * N + n[None, :]
    mask = (rows[:, None] < M) & (n[None, :] < N)
    x = tl.load(v + off, mask=mask, other=0.0).to(tl.float32)
    s = tl.sqrt(tl.sum(x * x, axis=1))
    gv = tl.load(g + rows * GS, mask=rows < M, other=0.0).to(tl.float32)
    if STORE_N:
        tl.store(nrm + rows, s, mask=rows < M)
    sc = gv / s
    tl.store(out + off, (x * sc[:, None]).to(out.dtype.element_ty), mask=mask)


@triton.jit
def _wn_d0_dot_norm_kernel(
    v,
    g,
    scale,
    nrm,
    M,
    N,
    R: tl.constexpr,
    C: tl.constexpr,
    GS: tl.constexpr,
    STORE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    ones = tl.full((C, 16), 1.0, dtype=tl.float32)
    acc = tl.zeros((R, 16), dtype=tl.float32)
    for c0 in range(0, N, C):
        cols = c0 + tl.arange(0, C)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(v + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        acc += tl.dot(x * x, ones, out_dtype=tl.float32, input_precision="ieee")
    s = tl.sqrt(tl.sum(acc, axis=1) * (1.0 / 16.0))
    gv = tl.load(g + rows * GS, mask=rows < M, other=0.0).to(tl.float32)
    tl.store(scale + rows, gv / s, mask=rows < M)
    if STORE_N:
        tl.store(nrm + rows, s, mask=rows < M)


@triton.jit
def _wn_d0_apply_row_kernel(v, scale, out, M, N, TOT, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    sc = tl.load(scale + r).to(tl.float32)
    base = r * N
    for c0 in range(0, N, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        a = base + cols
        mask = cols < N
        x = tl.load(v + a, mask=mask, other=0.0)
        tl.store(out + a, (x * sc.to(x.dtype)).to(out.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# dim == last
# --------------------------------------------------------------------------- #
@triton.jit
def _wn_dL_norm_kernel(v, scratch, M, K, R: tl.constexpr, C: tl.constexpr):
    """Column sums of squares for one column block, reduced over all rows."""
    kb = tl.program_id(0)
    cols = kb * C + tl.arange(0, C)
    cmask = cols < K
    ones = tl.full((16, R), 1.0, dtype=tl.float32)
    acc = tl.zeros((16, C), dtype=tl.float32)
    for r0 in range(0, M, R):
        rows = r0 + tl.arange(0, R)
        mask = (rows[:, None] < M) & cmask[None, :]
        x = tl.load(v + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        acc += tl.dot(ones, x * x, out_dtype=tl.float32, input_precision="ieee")
    tl.store(
        scratch
        + tl.arange(0, 16)[:, None] * (tl.num_programs(0) * C)
        + kb * C
        + tl.arange(0, C)[None, :],
        acc,
        mask=cmask[None, :],
    )


@triton.jit
def _wn_dL_part_kernel(
    v,
    part,
    M,
    K,
    R: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    GK: tl.constexpr,
):
    c = tl.program_id(0)
    kb = tl.program_id(1)
    cols = kb * C + tl.arange(0, C)
    cmask = cols < K
    rbase = c * BM
    nrow = tl.minimum(BM, M - rbase)
    ones = tl.full((16, R), 1.0, dtype=tl.float32)
    acc = tl.zeros((16, C), dtype=tl.float32)
    for r0 in range(0, nrow, R):
        rows = rbase + r0 + tl.arange(0, R)
        mask = (rows[:, None] < M) & cmask[None, :]
        x = tl.load(v + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        acc += tl.dot(ones, x * x, out_dtype=tl.float32, input_precision="ieee")
    offs = (
        (c * 16 + tl.arange(0, 16)[:, None]) * (GK * C)
        + kb * C
        + tl.arange(0, C)[None, :]
    )
    tl.store(part + offs, acc, mask=cmask[None, :])


@triton.jit
def _wn_dL_fin_kernel(
    part,
    scratch,
    NCH,
    NPAD: tl.constexpr,
    C: tl.constexpr,
    GK: tl.constexpr,
):
    kb = tl.program_id(0)
    cs = tl.arange(0, NPAD)
    offs = (cs[:, None] * 16) * (GK * C) + kb * C + tl.arange(0, C)[None, :]
    mask = (cs < NCH)[:, None]
    p = tl.load(part + offs, mask=mask, other=0.0)
    ones = tl.full((16, NPAD), 1.0, dtype=tl.float32)
    s = tl.dot(ones, p, out_dtype=tl.float32, input_precision="ieee")
    o2 = tl.arange(0, 16)[:, None] * (GK * C) + kb * C + tl.arange(0, C)[None, :]
    tl.store(scratch + o2, s)


@triton.jit
def _wn_dL_nrm_kernel(scratch, nrm, K, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < K
    cc = tl.minimum(cols, K - 1)
    n = tl.sqrt(tl.load(scratch + cc, mask=mask, other=1.0).to(tl.float32))
    tl.store(nrm + cc, n, mask=mask)


@triton.jit
def _wn_dL_apply_kernel(
    v, scratch, g, out, TOT, K: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < TOT
    grp = off % K
    ss = tl.load(scratch + grp, mask=mask, other=1.0).to(tl.float32)
    gv = tl.load(g + grp * GS, mask=mask, other=0.0).to(tl.float32)
    sc = gv / tl.sqrt(ss)
    x = tl.load(v + off, mask=mask, other=0.0)
    tl.store(out + off, (x * sc.to(x.dtype)).to(out.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# host wrappers
# --------------------------------------------------------------------------- #
def _pick_bs(M):
    for bs in (128, 64, 32, 16, 8, 4, 2, 1):
        if M % bs == 0:
            return bs
    return 1


def _forward_first(v, g, need_norm):
    M = v.shape[0]
    N = v.numel() // M
    out = torch.empty_like(v)
    gs = g.stride(0) if g.numel() > 1 else 0
    with torch_device_fn.device(v.device):
        if N == 1:
            nrm = (
                torch.empty(M, dtype=torch.float32, device=v.device)
                if need_norm
                else None
            )
            # scale1 kernel does not store norm; compute it only when needed
            _wn_d0_scale1_kernel[(triton.cdiv(M, 8192),)](
                v, g, out, M, GS=gs, BLOCK=8192, num_warps=4
            )
            if need_norm:
                _wn_d0_abs_kernel[(triton.cdiv(M, 8192),)](v, nrm, M, BLOCK=8192)
            return out, nrm
        if N <= 8192:
            BN = max(8, _npo2(N))
            BS = _pick_bs(M)
            nrm = (
                torch.empty(M, dtype=torch.float32, device=v.device)
                if need_norm
                else None
            )
            if BS >= 2:
                _wn_d0_multi_kernel[(triton.cdiv(M, BS),)](
                    v,
                    g,
                    out,
                    nrm if need_norm else out,
                    M,
                    N=N,
                    BN=BN,
                    BS=BS,
                    GS=gs,
                    STORE_N=need_norm,
                    num_warps=4,
                )
            else:
                _wn_d0_row_kernel[(M,)](
                    v,
                    g,
                    out,
                    nrm if need_norm else out,
                    M,
                    N=N,
                    BN=BN,
                    GS=gs,
                    STORE_N=need_norm,
                    num_warps=2,
                )
            return out, nrm
        # N > 8192: dot reduction + apply
        scale = torch.empty(M, dtype=torch.float32, device=v.device)
        nrm = (
            torch.empty(M, dtype=torch.float32, device=v.device) if need_norm else None
        )
        R = 64
        C = 256
        _wn_d0_dot_norm_kernel[(triton.cdiv(M, R),)](
            v,
            g,
            scale,
            nrm if need_norm else scale,
            M,
            N,
            R=R,
            C=C,
            GS=gs,
            STORE_N=need_norm,
            num_warps=4,
        )
        _wn_d0_apply_row_kernel[(M,)](
            v, scale, out, M, N, v.numel(), BLOCK=8192, num_warps=4
        )
        return out, nrm


def _forward_last(v, g, need_norm):
    K = v.shape[-1]
    M = v.numel() // K
    out = torch.empty_like(v)
    gs = g.stride(v.dim() - 1) if g.numel() > 1 else 0
    C = max(16, min(256, _npo2(K)))
    GK = triton.cdiv(K, C)
    scratch = torch.empty(16 * GK * C, dtype=torch.float32, device=v.device)
    nrm = torch.empty(K, dtype=torch.float32, device=v.device) if need_norm else None
    R = max(16, min(64, _npo2(M)))
    with torch_device_fn.device(v.device):
        if M <= 8192:
            _wn_dL_norm_kernel[(GK,)](v, scratch, M, K, R=R, C=C, num_warps=4)
        else:
            BM = R * triton.cdiv(triton.cdiv(M, triton.cdiv(M, 8192)), R)
            NCH = triton.cdiv(M, BM)
            NPAD = max(16, _npo2(NCH))
            part = torch.empty(NPAD * 16 * GK * C, dtype=torch.float32, device=v.device)
            _wn_dL_part_kernel[(NCH, GK)](
                v, part, M, K, R=R, C=C, BM=BM, GK=GK, num_warps=4
            )
            _wn_dL_fin_kernel[(GK,)](
                part, scratch, NCH, NPAD=NPAD, C=C, GK=GK, num_warps=4
            )
        if need_norm:
            _wn_dL_nrm_kernel[(triton.cdiv(K, 1024),)](
                scratch, nrm, K, BLOCK=1024, num_warps=4
            )
        tot = v.numel()
        blk = 8192 if tot >= 65536 else 2048
        _wn_dL_apply_kernel[(triton.cdiv(tot, blk),)](
            v, scratch, g, out, tot, K=K, GS=gs, BLOCK=blk, num_warps=4
        )
    return out, nrm


@triton.jit
def _wn_d0_abs_kernel(v, nrm, M, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < M
    ic = tl.minimum(i, M - 1)
    x = tl.load(v + ic, mask=mask, other=0.0).to(tl.float32)
    tl.store(nrm + ic, tl.abs(x), mask=mask)


class _WeightNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim):
        d = dim % v.dim()
        try:
            need = bool(ctx.needs_input_grad[0] or ctx.needs_input_grad[1])
        except Exception:
            need = True
        if v.numel() == 0:
            out = torch.empty_like(v)
            nrm = (
                torch.empty(v.shape[d], dtype=torch.float32, device=v.device)
                if need
                else None
            )
            return out
        vc = v if v.is_contiguous() else v.contiguous()
        gc = g if g.is_contiguous() else g.contiguous()
        if d == 0:
            out, nrm = _forward_first(vc, gc, need)
            ctx.can_fuse = True
        elif d == v.dim() - 1:
            out, nrm = _forward_last(vc, gc, need)
            ctx.can_fuse = True
        else:
            out, nrm = weight_norm_except_dim(vc, gc, d)
            ctx.can_fuse = False
        ctx.dim = d
        if need:
            ctx.save_for_backward(vc, gc, nrm)
        return out

    @staticmethod
    def backward(ctx, grad):
        v, g, nrm = ctx.saved_tensors
        if grad.is_contiguous():
            gr = grad
        else:
            gr = grad.contiguous()
        if ctx.can_fuse:
            v_grad, g_grad = weight_norm_interface_backward(gr, v, g, nrm, ctx.dim)
        else:
            v_grad, g_grad = weight_norm_except_dim_backward(gr, v, g, nrm, ctx.dim)
        if v_grad.shape != v.shape:
            v_grad = v_grad.view(v.shape)
        return v_grad, g_grad, None


def _weight_norm(v: torch.Tensor, g: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """ATen entry point for the Kunlunxin weight-normalization path."""
    logger.debug("GEMS_KUNLUNXIN _WEIGHT_NORM")
    if v.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"_weight_norm does not support {v.dtype}")
    return _WeightNormFn.apply(v, g, dim)
