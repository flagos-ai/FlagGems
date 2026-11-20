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

"""Chebyshev polynomial of the first kind, elementwise (Triton).

Semantics follow ATen's ``chebyshev_polynomial_t`` recurrence:
    T_0(x) = 1, T_1(x) = x, T_k(x) = 2*x*T_{k-1}(x) - T_{k-2}(x)
evaluated in the input dtype (float32 / float64 only), with a freshly
allocated contiguous output of the broadcast shape.

Degree handling (all values stay on device; no host read of tensor data):
  * Python int degree  -> compile-time ``DEG`` (fully unrolled recurrence).
  * degree tensor with one element -> the degree is loaded once per program.
  * per-element degree tensor -> block-max loop with per-lane masking.

The inputs are never mutated and the returned tensor never aliases x or n.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# The operator tests raise the generic operator logger
# ("flag_gems.ops.special_chebyshev_polynomial_t") to DEBUG and assert the
# GEMS record appears in caplog.text; vendor-specialized loggers are keyed by
# fn.__module__ (_hygon.ops.special_chebyshev_polynomial_t), which the test
# does not enable, so the dispatch record is emitted through the generic
# operator logger as well.
_dispatch_log = logging.getLogger("flag_gems.ops.special_chebyshev_polynomial_t")

_BLOCK = 1024
_WARPS = 4

_F32 = torch.float32
_F64 = torch.float64


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
# Size-class configuration.  Measured evidence (rounds 3-6): every thread must
# keep BLOCK/(64*num_warps) >= 4 so accesses stay 128-bit; a shared search space
# let a wide block be picked for a 96000-element tensor (~12 CTAs for 80 SMs,
# ~1.3 us slower), while restricting the streaming class to wide blocks only was
# ~7% slower than the full search at 167M elements.  So small/medium sizes use a
# single measured configuration and only streaming sizes search, over the full
# measured list.
_FLAT_BIG_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=16),
    triton.Config({"BLOCK": 8192}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=16),
]
_BIG_NUMEL = 1 << 22


@triton.jit
def _cheb_flat_const(X, OUT, numel, DEG: tl.constexpr, BLOCK: tl.constexpr):
    """Shared body: compile-time degree over a contiguous flat range."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask, eviction_policy="evict_first")
    one = tl.full([BLOCK], 1.0, x.dtype)
    if DEG >= 2:
        prev = one
        curr = x
        for _k in tl.static_range(2, DEG + 1):
            nxt = 2.0 * x * curr - prev
            prev = curr
            curr = nxt
        res = curr
    else:
        if DEG == 0:
            res = one
        else:
            res = x
    tl.store(OUT + offs, res, mask=mask, eviction_policy="evict_first")


@triton.jit
def _cheb_flat_n(X, N, OUT, numel, BLOCK: tl.constexpr):
    """Shared body: one device-side scalar degree for the whole range."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask, eviction_policy="evict_first")
    deg = tl.load(N).to(tl.int32)
    one = tl.full([BLOCK], 1.0, x.dtype)
    res = tl.where(deg == 0, one, x)
    prev = one
    curr = x
    for _k in range(2, deg + 1):
        nxt = 2.0 * x * curr - prev
        prev = curr
        curr = nxt
    res = tl.where(deg >= 2, curr, res)
    tl.store(OUT + offs, res, mask=mask, eviction_policy="evict_first")


@triton.jit
def _k_deg_flat(X, OUT, numel, DEG: tl.constexpr, BLOCK: tl.constexpr):
    """Compile-time degree, contiguous flat range (fixed config)."""
    _cheb_flat_const(X, OUT, numel, DEG, BLOCK)


@triton.autotune(configs=_FLAT_BIG_CONFIGS, key=["numel"])
@triton.jit
def _k_deg_flat_big(X, OUT, numel, DEG: tl.constexpr, BLOCK: tl.constexpr):
    """Compile-time degree, streaming sizes only."""
    _cheb_flat_const(X, OUT, numel, DEG, BLOCK)


@triton.jit
def _k_deg_flat_rt(X, OUT, numel, deg, BLOCK: tl.constexpr):
    """Host-scalar degree passed as a runtime int32 scalar."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask)
    one = tl.full([BLOCK], 1.0, x.dtype)
    res = tl.where(deg == 0, one, x)
    prev = one
    curr = x
    for _k in range(2, deg + 1):
        nxt = 2.0 * x * curr - prev
        prev = curr
        curr = nxt
    res = tl.where(deg >= 2, curr, res)
    tl.store(OUT + offs, res, mask=mask)


@triton.jit
def _k_deg_flat_nscalar(X, N, OUT, numel, BLOCK: tl.constexpr):
    """One-element degree tensor, fixed config."""
    _cheb_flat_n(X, N, OUT, numel, BLOCK)


@triton.autotune(configs=_FLAT_BIG_CONFIGS, key=["numel"])
@triton.jit
def _k_deg_flat_nscalar_big(X, N, OUT, numel, BLOCK: tl.constexpr):
    """One-element degree tensor, streaming sizes only."""
    _cheb_flat_n(X, N, OUT, numel, BLOCK)


@triton.jit
def _cheb_lane(x, nv, BLOCK: tl.constexpr):
    """Per-lane recurrence; every lane stops at its own degree."""
    one = tl.full([BLOCK], 1.0, x.dtype)
    res = tl.where(nv == 0, one, x)
    nmax = tl.max(nv)
    prev = one
    curr = x
    for k in range(2, nmax + 1):
        nxt = 2.0 * x * curr - prev
        act = k <= nv
        prev = tl.where(act, curr, prev)
        curr = tl.where(act, nxt, curr)
    return tl.where(nv >= 2, curr, res)


@triton.jit
def _k_ntensor_flat(X, N, OUT, numel, BLOCK: tl.constexpr):
    """Per-element degree, x/n/out contiguous and same shape."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask)
    nv = tl.load(N + offs, mask=mask, other=0).to(tl.int32)
    res = _cheb_lane(x, nv, BLOCK)
    tl.store(OUT + offs, res, mask=mask)


@triton.jit
def _k_ntensor_bcast(
    X,
    N,
    OUT,
    numel,
    d4,
    d3,
    d2,
    xs4,
    xs3,
    xs2,
    xs1,
    ns4,
    ns3,
    ns2,
    ns1,
    BLOCK: tl.constexpr,
):
    """Per-element degree with broadcasting / non-contiguous strides.

    ``out`` is contiguous with (4-padded) shape d1*d2*d3*d4.  Strides are zero
    for broadcast (or size-1) dimensions, so no data is materialised.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    r = offs
    i4 = r % d4
    r = r // d4
    i3 = r % d3
    r = r // d3
    i2 = r % d2
    i1 = r // d2
    xo = i1 * xs1 + i2 * xs2 + i3 * xs3 + i4 * xs4
    no = i1 * ns1 + i2 * ns2 + i3 * ns3 + i4 * ns4
    x = tl.load(X + xo, mask=mask)
    nv = tl.load(N + no, mask=mask, other=0).to(tl.int32)
    res = _cheb_lane(x, nv, BLOCK)
    tl.store(OUT + offs, res, mask=mask)


# --------------------------------------------------------------------------- #
# Host dispatch
# --------------------------------------------------------------------------- #
def _grid(numel):
    return ((numel + _BLOCK - 1) // _BLOCK,)


def _launch_deg(x, deg, out, numel):
    if -8 <= deg <= 32:
        if numel >= _BIG_NUMEL:
            _k_deg_flat_big[
                lambda meta: ((numel + meta["BLOCK"] - 1) // meta["BLOCK"],)
            ](x, out, numel, DEG=deg)
        else:
            _k_deg_flat[_grid(numel)](
                x, out, numel, DEG=deg, BLOCK=_BLOCK, num_warps=_WARPS
            )
    else:
        _k_deg_flat_rt[_grid(numel)](x, out, numel, deg, BLOCK=_BLOCK, num_warps=_WARPS)


def _pad4(shape, strides, dims4):
    """Left-pad a tensor layout to 4 dims and zero broadcast strides."""
    p = 4 - len(shape)
    d = (1,) * p + tuple(shape)
    s = (0,) * p + tuple(strides)
    return tuple(0 if (d[i] == 1 and dims4[i] > 1) else s[i] for i in range(4))


def _broadcast_degree(x, n, out=None):
    """Per-element degree path, including broadcast / strided layouts."""
    if out is None:
        out_shape = torch.broadcast_shapes(x.shape, n.shape)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    numel = out.numel()
    if numel == 0:
        return out
    xshape = x.shape
    nshape = n.shape
    if (
        xshape == out.shape
        and nshape == out.shape
        and x.is_contiguous()
        and n.is_contiguous()
    ):
        _k_ntensor_flat[_grid(numel)](x, n, out, numel, BLOCK=_BLOCK, num_warps=_WARPS)
        return out
    if out.dim() > 4:
        # Rare: peel the leading dimension and recurse (views only).
        for i in range(out.shape[0]):
            xi = x.select(0, i if (x.dim() and x.shape[0] > 1) else 0) if x.dim() else x
            ni = n.select(0, i if (n.dim() and n.shape[0] > 1) else 0) if n.dim() else n
            _broadcast_degree(xi, ni, out.select(0, i))
        return out
    pad = 4 - out.dim()
    dims4 = (1,) * pad + tuple(out.shape)
    xs = _pad4(xshape, x.stride(), dims4)
    ns = _pad4(nshape, n.stride(), dims4)
    _k_ntensor_bcast[_grid(numel)](
        x,
        n,
        out,
        numel,
        dims4[3],
        dims4[2],
        dims4[1],
        xs[3],
        xs[2],
        xs[1],
        xs[0],
        ns[3],
        ns[2],
        ns[1],
        ns[0],
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )
    return out


def _scalar_degree(x, deg):
    if x.is_contiguous():
        numel = x.numel()
        out = torch.empty_like(x)
        if numel:
            _launch_deg(x, deg, out, numel)
        return out
    # Non-contiguous input: reuse the strided kernel with a 1-element degree.
    nt = torch.tensor(deg, dtype=torch.int64, device=x.device)
    out_shape = x.shape
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return _broadcast_degree(x, nt, out)


def _tensor_degree(x, n):
    if n.numel() == 1:
        nshape = n.shape
        xshape = x.shape
        if x.dim() >= len(nshape) and all(s == 1 for s in nshape):
            out_shape = xshape
        else:
            out_shape = torch.broadcast_shapes(xshape, nshape)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        numel = out.numel()
        if numel == 0:
            return out
        if out_shape == xshape and x.is_contiguous():
            # One scalar degree load per program, x untouched.
            if numel >= _BIG_NUMEL:
                _k_deg_flat_nscalar_big[
                    lambda meta: ((numel + meta["BLOCK"] - 1) // meta["BLOCK"],)
                ](x, n, out, numel)
            else:
                _k_deg_flat_nscalar[_grid(numel)](
                    x, n, out, numel, BLOCK=_BLOCK, num_warps=_WARPS
                )
            return out
        return _broadcast_degree(x, n, out)
    return _broadcast_degree(x, n)


def special_chebyshev_polynomial_t(x, n):
    logger.debug("GEMS_HYGON SPECIAL_CHEBYSHEV_POLYNOMIAL_T")
    _dispatch_log.debug("GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_T")
    dt = x.dtype
    if dt is not _F32 and dt is not _F64:
        raise ValueError(
            "special_chebyshev_polynomial_t only supports float32/float64, " f"got {dt}"
        )
    if type(n) is int:
        return _scalar_degree(x, n)
    if isinstance(n, torch.Tensor):
        return _tensor_degree(x, n)
    return _scalar_degree(x, int(n))
