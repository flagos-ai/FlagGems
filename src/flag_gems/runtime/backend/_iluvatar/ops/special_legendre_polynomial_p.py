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


# ---------------------------------------------------------------------------
# Fast path: scalar polynomial degree n, contiguous x.
# N is a compile-time constant so the Legendre recurrence is fully unrolled
# and folded at compile time; the kernel is then pure load/compute/store.
# ---------------------------------------------------------------------------
@triton.jit
def _legendre_scalar_flat(
    x_ptr,
    out_ptr,
    numel,
    N: tl.constexpr,
    CD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(CD)

    if N == 0:
        res = tl.full([BLOCK], 1.0, CD)
    elif N == 1:
        res = x
    else:
        pjm2 = tl.full([BLOCK], 1.0, CD)  # P_0
        pjm1 = x  # P_1
        for j in tl.static_range(2, N + 1):
            # j is a Python int (constexpr); fold the constants directly
            c1 = 2.0 * j - 1.0
            c2 = j - 1.0
            c3 = 1.0 / j
            pj = (c1 * x * pjm1 - c2 * pjm2) * c3
            pjm2 = pjm1
            pjm1 = pj
        res = pjm1

    tl.store(
        out_ptr + offs, res.to(x_ptr.dtype.element_ty), mask=mask, cache_modifier=".cg"
    )


# ---------------------------------------------------------------------------
# Generic path: tensor degree n, or non-contiguous / broadcast x and n.
# Multi-dim strides are passed as separate constexpr scalars (no constexpr
# tuple indexing), and the degree loop runs up to the block-wise max of n.
# ---------------------------------------------------------------------------
@triton.jit
def _legendre_generic(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    NS2: tl.constexpr,
    NDIM: tl.constexpr,
    CD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    if NDIM == 1:
        x_off = offs * XS0
        n_off = offs * NS0
    elif NDIM == 2:
        s1 = offs % S1
        s0 = offs // S1
        x_off = s0 * XS0 + s1 * XS1
        n_off = s0 * NS0 + s1 * NS1
    else:
        s2 = offs % S2
        r = offs // S2
        s1 = r % S1
        s0 = r // S1
        x_off = s0 * XS0 + s1 * XS1 + s2 * XS2
        n_off = s0 * NS0 + s1 * NS1 + s2 * NS2

    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(CD)
    n = tl.load(n_ptr + n_off, mask=mask, other=0)
    nf = n.to(tl.int64)

    one = tl.full([BLOCK], 1.0, CD)
    pjm2 = one
    pjm1 = x
    max_n = tl.max(nf, axis=0)

    for j in range(2, max_n + 1):
        jf = j.to(CD)
        active = nf >= j
        pj = ((2.0 * jf - 1.0) * x * pjm1 - (jf - 1.0) * pjm2) / jf
        pjm2 = tl.where(active, pjm1, pjm2)
        pjm1 = tl.where(active, pj, pjm1)

    res = tl.where(nf == 0, one, pjm1)
    tl.store(
        out_ptr + offs, res.to(x_ptr.dtype.element_ty), mask=mask, cache_modifier=".cg"
    )


# Launch geometry measured on target (do_bench): 2 elements/thread
# (BLOCK=1024/num_warps=16) sustains ~610 GB/s vs ~570 GB/s at 4 elems/thread
# for large tensors; store cache_modifier .cg (L1-bypass streaming write)
# adds ~1% more. For tiny tensors fewer, wider blocks (BLOCK=1024/w8)
# minimize launch/scheduling overhead.
_BLOCK = 1024
_NUM_WARPS = 16
_SMALL_LIMIT = 8192
_SMALL_BLOCK = 2048
_SMALL_NUM_WARPS = 32


def _config(numel):
    if numel <= _SMALL_LIMIT:
        return _SMALL_BLOCK, _SMALL_NUM_WARPS
    return _BLOCK, _NUM_WARPS


def _compute_dtype(x):
    if x.dtype == torch.float64:
        return tl.float64
    if x.dtype == torch.float32:
        return tl.float32
    if x.dtype == torch.float16:
        return tl.float16
    if x.dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


def _is_scalar_n(n):
    if isinstance(n, (int, float)):
        return True
    if isinstance(n, torch.Tensor):
        return n.dim() == 0 and n.numel() == 1
    return False


def _scalar_n_value(n):
    if isinstance(n, (int, float)):
        return int(n)
    return int(n.item())


def special_legendre_polynomial_p(x, n):
    cdtype = _compute_dtype(x)

    if _is_scalar_n(n):
        d = _scalar_n_value(n)
        if d < 0:
            d = 0
        if x.is_contiguous():
            numel = x.numel()
            out = torch.empty_like(x)
            blk, warps = _config(numel)
            grid = (triton.cdiv(numel, blk),)
            _legendre_scalar_flat[grid](
                x,
                out,
                numel,
                N=d,
                CD=cdtype,
                BLOCK=blk,
                num_warps=warps,
            )
            return out
        # non-contiguous x: fall through to generic path with n materialized
        n = torch.as_tensor(d, device=x.device, dtype=torch.int64)

    n_t = n if isinstance(n, torch.Tensor) else torch.as_tensor(n, device=x.device)
    if n_t.dim() == 0:
        n_t = n_t.reshape(1)

    shape = tuple(torch.broadcast_shapes(tuple(x.shape), tuple(n_t.shape)))
    xb, nb = torch.broadcast_tensors(x, n_t)
    numel = 1
    for s in shape:
        numel *= s
    out = torch.empty(shape, dtype=x.dtype, device=x.device)

    ndim = len(shape)
    s0 = shape[0] if ndim >= 1 else 1
    s1 = shape[1] if ndim >= 2 else 1
    s2 = shape[2] if ndim >= 3 else 1
    xs0 = xb.stride()[0] if ndim >= 1 else 1
    xs1 = xb.stride()[1] if ndim >= 2 else 1
    xs2 = xb.stride()[2] if ndim >= 3 else 1
    ns0 = nb.stride()[0] if ndim >= 1 else 1
    ns1 = nb.stride()[1] if ndim >= 2 else 1
    ns2 = nb.stride()[2] if ndim >= 3 else 1

    blk, warps = _config(numel)
    grid = (triton.cdiv(numel, blk),)
    _legendre_generic[grid](
        xb,
        nb,
        out,
        numel,
        S0=s0,
        S1=s1,
        S2=s2,
        XS0=xs0,
        XS1=xs1,
        XS2=xs2,
        NS0=ns0,
        NS1=ns1,
        NS2=ns2,
        NDIM=ndim,
        CD=cdtype,
        BLOCK=blk,
        num_warps=warps,
    )
    return out
