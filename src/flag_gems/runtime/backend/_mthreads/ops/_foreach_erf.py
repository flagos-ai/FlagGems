"""Triton implementation of aten::_foreach_erf.

Applies erf elementwise to every tensor of the input sequence, preserving list
order, shape, dtype and device, returning freshly allocated outputs.

Math: erf is computed in float32 opmath for 16-bit inputs (ATen parity: one
rounding on store), natively for float32/float64, using the backend libdevice
erf via tl.math.erf.

Dispatch:
  * empty sequence            -> RuntimeError (ATen foreach contract)
  * integral / bool tensors    -> promoted to the default float dtype
  * complex / other dtypes     -> RuntimeError (unsupported)
  * homogeneous contiguous list of floating tensors (same dtype and numel) is
    served by a single fused launch over a 2-D grid (block, tensor index),
    with the per-CTA tensor pointer chosen by a balanced 4-level select tree;
    this removes the per-launch host cost that dominates small-tensor lists and
    wins up to 2^28 elements per tensor.  Giant per-tensor sizes keep the 1-D
    per-tensor launch, where a multi-million-CTA 2-D grid measurably loses
    bandwidth.
  * otherwise one launch per tensor (flat kernel when contiguous, generic
    RANK-decomposed strided kernel otherwise); zero-numel tensors never launch
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_FUSED_BLOCK_BIG = 4096  # used only when the fused grid still fills all SMs
_FUSED_WARPS_BIG = 8
_FUSED_MIN_CTAS = 60  # SMs on the target device
_NUM_WARPS = 4
_LONG_LIMIT = 2147483648  # 2**31
_MAX_FUSE = 16
_FUSE_MAX_ELEMS = 268435456  # 2**28: above this a 2-D fused grid loses to 1-D launches


@triton.jit
def _erf_flat(x_ptr, o_ptr, n, BLOCK: tl.constexpr, LONG: tl.constexpr):
    pid = tl.program_id(0)
    if LONG:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if v.dtype == tl.float64:
        r = tl.math.erf(v)
    else:
        r = tl.math.erf(v.to(tl.float32))
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _erf_fused(
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    x4,
    o4,
    x5,
    o5,
    x6,
    o6,
    x7,
    o7,
    x8,
    o8,
    x9,
    o9,
    x10,
    o10,
    x11,
    o11,
    x12,
    o12,
    x13,
    o13,
    x14,
    o14,
    x15,
    o15,
    n,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    k = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xp = x0
    op = o0
    # Balanced 4-level select tree: every CTA picks its tensor from a uniform
    # scalar with a short dependency chain (a 15-step serial chain measurably
    # lengthens the path to the first load at a 16 B/thread payload).
    if k < 8:
        if k < 4:
            if k < 2:
                if k == 0:
                    xp = x0
                    op = o0
                else:
                    xp = x1
                    op = o1
            else:
                if k == 2:
                    xp = x2
                    op = o2
                else:
                    xp = x3
                    op = o3
        else:
            if k < 6:
                if k == 4:
                    xp = x4
                    op = o4
                else:
                    xp = x5
                    op = o5
            else:
                if k == 6:
                    xp = x6
                    op = o6
                else:
                    xp = x7
                    op = o7
    else:
        if k < 12:
            if k < 10:
                if k == 8:
                    xp = x8
                    op = o8
                else:
                    xp = x9
                    op = o9
            else:
                if k == 10:
                    xp = x10
                    op = o10
                else:
                    xp = x11
                    op = o11
        else:
            if k < 14:
                if k == 12:
                    xp = x12
                    op = o12
                else:
                    xp = x13
                    op = o13
            else:
                if k == 14:
                    xp = x14
                    op = o14
                else:
                    xp = x15
                    op = o15
    v = tl.load(xp + offs, mask=mask, other=0.0)
    if v.dtype == tl.float64:
        r = tl.math.erf(v)
    else:
        r = tl.math.erf(v.to(tl.float32))
    tl.store(op + offs, r.to(v.dtype), mask=mask)


@triton.jit
def _erf_strided(
    x_ptr,
    o_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rem = pid * BLOCK + tl.arange(0, BLOCK)
    mask = rem < n
    xi = tl.zeros((BLOCK,), dtype=tl.int32)
    oi = tl.zeros((BLOCK,), dtype=tl.int32)
    if RANK > 7:
        c = rem % d7
        rem = rem // d7
        xi += c * s7
        oi += c * t7
    if RANK > 6:
        c = rem % d6
        rem = rem // d6
        xi += c * s6
        oi += c * t6
    if RANK > 5:
        c = rem % d5
        rem = rem // d5
        xi += c * s5
        oi += c * t5
    if RANK > 4:
        c = rem % d4
        rem = rem // d4
        xi += c * s4
        oi += c * t4
    if RANK > 3:
        c = rem % d3
        rem = rem // d3
        xi += c * s3
        oi += c * t3
    if RANK > 2:
        c = rem % d2
        rem = rem // d2
        xi += c * s2
        oi += c * t2
    if RANK > 1:
        c = rem % d1
        rem = rem // d1
        xi += c * s1
        oi += c * t1
    if RANK > 0:
        c = rem % d0
        rem = rem // d0
        xi += c * s0
        oi += c * t0
    v = tl.load(x_ptr + xi, mask=mask, other=0.0)
    if v.dtype == tl.float64:
        r = tl.math.erf(v)
    else:
        r = tl.math.erf(v.to(tl.float32))
    tl.store(o_ptr + oi, r.to(o_ptr.dtype.element_ty), mask=mask)


def _launch_one(x, out, n):
    if x.is_contiguous():
        _erf_flat[((n + (_BLOCK - 1)) // _BLOCK,)](
            x, out, n, BLOCK=_BLOCK, LONG=n >= _LONG_LIMIT, num_warps=_NUM_WARPS
        )
    else:
        rank = x.dim()
        dims = list(x.shape) + [1] * (8 - rank)
        sstr = list(x.stride()) + [0] * (8 - rank)
        ostr = list(out.stride()) + [0] * (8 - rank)
        _erf_strided[((n + (_BLOCK - 1)) // _BLOCK,)](
            x,
            out,
            n,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            dims[4],
            dims[5],
            dims[6],
            dims[7],
            sstr[0],
            sstr[1],
            sstr[2],
            sstr[3],
            sstr[4],
            sstr[5],
            sstr[6],
            sstr[7],
            ostr[0],
            ostr[1],
            ostr[2],
            ostr[3],
            ostr[4],
            ostr[5],
            ostr[6],
            ostr[7],
            RANK=rank,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )


def _erf_one(x):
    n = x.numel()
    dt = x.dtype
    if dt.is_floating_point:
        out = torch.empty_like(x)
    elif not dt.is_complex:
        out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    else:
        raise RuntimeError("erf not implemented for '%s'" % dt)
    if n == 0:
        return out
    _launch_one(x, out, n)
    return out


def _run_fused(ts, n):
    outs = [torch.empty_like(t) for t in ts]
    k = len(ts)
    # Per-thread payload vs CTA count: a bigger block amortises the per-CTA
    # pointer select over more bytes, but it must not shrink the whole grid
    # below one CTA per SM, which is what the launch-bound tiny lists are
    # limited by.  The large variant keeps the same 16 elements/thread as the
    # small one by also doubling the warps per CTA.
    if n >= _FUSED_BLOCK_BIG * _FUSED_MIN_CTAS:
        block = _FUSED_BLOCK_BIG
        warps = _FUSED_WARPS_BIG
    else:
        block = _BLOCK
        warps = _NUM_WARPS
    grid = ((n + (block - 1)) // block, k)
    if k == _MAX_FUSE:
        _erf_fused[grid](
            ts[0],
            outs[0],
            ts[1],
            outs[1],
            ts[2],
            outs[2],
            ts[3],
            outs[3],
            ts[4],
            outs[4],
            ts[5],
            outs[5],
            ts[6],
            outs[6],
            ts[7],
            outs[7],
            ts[8],
            outs[8],
            ts[9],
            outs[9],
            ts[10],
            outs[10],
            ts[11],
            outs[11],
            ts[12],
            outs[12],
            ts[13],
            outs[13],
            ts[14],
            outs[14],
            ts[15],
            outs[15],
            n,
            K=k,
            BLOCK=block,
            num_warps=warps,
        )
    else:
        xx = list(ts) + [ts[0]] * (_MAX_FUSE - k)
        oo = outs + [outs[0]] * (_MAX_FUSE - k)
        _erf_fused[grid](
            xx[0],
            oo[0],
            xx[1],
            oo[1],
            xx[2],
            oo[2],
            xx[3],
            oo[3],
            xx[4],
            oo[4],
            xx[5],
            oo[5],
            xx[6],
            oo[6],
            xx[7],
            oo[7],
            xx[8],
            oo[8],
            xx[9],
            oo[9],
            xx[10],
            oo[10],
            xx[11],
            oo[11],
            xx[12],
            oo[12],
            xx[13],
            oo[13],
            xx[14],
            oo[14],
            xx[15],
            oo[15],
            n,
            K=k,
            BLOCK=block,
            num_warps=warps,
        )
    return outs


def _foreach_erf(self, *extra, **kwargs):
    tensors = self if not extra else (self,) + extra
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    count = len(tensors)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if count > 1:
        first = tensors[0]
        dtype = first.dtype
        n = first.numel()
        if (
            0 < n <= _FUSE_MAX_ELEMS
            and dtype.is_floating_point
            and first.is_contiguous()
        ):
            homogeneous = True
            for t in tensors:
                if t.dtype != dtype or t.numel() != n or not t.is_contiguous():
                    homogeneous = False
                    break
            if homogeneous:
                if count <= _MAX_FUSE:
                    return _run_fused(tensors, n)
                outs = []
                for start in range(0, count, _MAX_FUSE):
                    outs.extend(_run_fused(tensors[start : start + _MAX_FUSE], n))
                return outs
    return [_erf_one(t) for t in tensors]
