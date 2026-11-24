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

"""Chebyshev polynomial of the third kind V_n(x) (Gems / Triton candidate).

V_0(x) = 1, V_1(x) = 2x - 1, V_n(x) = 2x V_{n-1}(x) - V_{n-2}(x).

The integer three-term recurrence is used instead of the cos/acos closed form:
it needs only FMAs plus selects (no transcendental work) and it is
well-conditioned on the whole real line, including x = -1.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# The gems dispatch test captures records on the reference operator's logger, so
# the dispatch trace must be emitted through that exact logger name.
_dispatch_logger = logging.getLogger("flag_gems.ops.special_chebyshev_polynomial_v")

_BLOCK = 1024
_WARPS = 4
_MAX_RANK = 8


@triton.jit
def _v_eval(xf, ni, two_x):
    """Evaluate V_n(x) for integer degrees ni, accumulating in xf.dtype."""
    y = two_x - 1.0
    one = two_x * 0.0 + 1.0
    res = tl.where(ni == 0, one, y)
    p0 = one
    p1 = y
    p2 = two_x * p1 - p0
    res = tl.where(ni == 2, p2, res)
    p0 = p1
    p1 = p2
    p2 = two_x * p1 - p0
    res = tl.where(ni == 3, p2, res)
    p0 = p1
    p1 = p2
    p2 = two_x * p1 - p0
    res = tl.where(ni == 4, p2, res)
    p0 = p1
    p1 = p2
    # Degrees above the common small range: keep going until the tile maximum.
    nmax = tl.max(ni, axis=0)
    for k in range(5, nmax + 1):
        p2 = two_x * p1 - p0
        res = tl.where(ni == k, p2, res)
        p0 = p1
        p1 = p2
    return res


@triton.jit
def _chev_v_flat(
    x_ptr,
    n_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    N_SCALAR: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    if FP64:
        xf = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float64)
    else:
        xf = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if N_SCALAR:
        ni = tl.load(n_ptr).to(tl.float32)
        ni = tl.floor(ni + 0.5).to(tl.int32) + tl.zeros([BLOCK], tl.int32)
    else:
        nraw = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.float32)
        ni = tl.floor(nraw + 0.5).to(tl.int32)
    res = _v_eval(xf, ni, xf * 2.0)
    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _chev_v_strided(
    x_ptr,
    n_ptr,
    out_ptr,
    N,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    ns0,
    ns1,
    ns2,
    ns3,
    ns4,
    ns5,
    ns6,
    ns7,
    BLOCK: tl.constexpr,
    N_SCALAR: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    idx = offs
    xoff = tl.zeros([BLOCK], tl.int32)
    noff = tl.zeros([BLOCK], tl.int32)
    c = idx % sh0
    idx = idx // sh0
    xoff += c * xs0
    noff += c * ns0
    c = idx % sh1
    idx = idx // sh1
    xoff += c * xs1
    noff += c * ns1
    c = idx % sh2
    idx = idx // sh2
    xoff += c * xs2
    noff += c * ns2
    c = idx % sh3
    idx = idx // sh3
    xoff += c * xs3
    noff += c * ns3
    c = idx % sh4
    idx = idx // sh4
    xoff += c * xs4
    noff += c * ns4
    c = idx % sh5
    idx = idx // sh5
    xoff += c * xs5
    noff += c * ns5
    c = idx % sh6
    idx = idx // sh6
    xoff += c * xs6
    noff += c * ns6
    c = idx % sh7
    xoff += c * xs7
    noff += c * ns7

    if FP64:
        xf = tl.load(x_ptr + xoff, mask=mask, other=0.0).to(tl.float64)
    else:
        xf = tl.load(x_ptr + xoff, mask=mask, other=0.0).to(tl.float32)
    if N_SCALAR:
        ni = tl.load(n_ptr).to(tl.float32)
        ni = tl.floor(ni + 0.5).to(tl.int32) + tl.zeros([BLOCK], tl.int32)
    else:
        nraw = tl.load(n_ptr + noff, mask=mask, other=0).to(tl.float32)
        ni = tl.floor(nraw + 0.5).to(tl.int32)
    res = _v_eval(xf, ni, xf * 2.0)
    tl.store(out_ptr + offs, res, mask=mask)


def _broadcast_strides(tensor, shape):
    """Effective strides of ``tensor`` after broadcasting to ``shape``."""
    nd = len(shape)
    strides = [0] * nd
    offset = nd - tensor.dim()
    tshape = tensor.shape
    for i in range(tensor.dim()):
        if tshape[i] != 1:
            strides[offset + i] = tensor.stride(i)
    return strides


def _is_dense(strides, shape):
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] != 1:
            if strides[i] != acc:
                return False
            acc *= shape[i]
    return True


def special_chebyshev_polynomial_v(x, n):
    _dispatch_logger.debug("GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_V")
    x_shape = x.shape
    n_shape = n.shape
    if x_shape == n_shape:
        out = torch.empty_like(x)
        total = out.numel()
        if total == 0:
            return out
        if x.is_contiguous() and n.is_contiguous():
            _chev_v_flat[((total + _BLOCK - 1) // _BLOCK,)](
                x,
                n,
                out,
                total,
                BLOCK=_BLOCK,
                N_SCALAR=False,
                FP64=(x.dtype == torch.float64),
                num_warps=_WARPS,
            )
            return out
        out_shape = x_shape
    else:
        out_shape = torch.broadcast_shapes(x_shape, n_shape)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        total = out.numel()
        if total == 0:
            return out
        if x.numel() == total and x.is_contiguous():
            if n.numel() == 1:
                _chev_v_flat[((total + _BLOCK - 1) // _BLOCK,)](
                    x,
                    n,
                    out,
                    total,
                    BLOCK=_BLOCK,
                    N_SCALAR=True,
                    FP64=(x.dtype == torch.float64),
                    num_warps=_WARPS,
                )
                return out
            nstrides = _broadcast_strides(n, out_shape)
            if _is_dense(nstrides, out_shape):
                _chev_v_flat[((total + _BLOCK - 1) // _BLOCK,)](
                    x,
                    n,
                    out,
                    total,
                    BLOCK=_BLOCK,
                    N_SCALAR=False,
                    FP64=(x.dtype == torch.float64),
                    num_warps=_WARPS,
                )
                return out

    nd = len(out_shape)
    if nd > _MAX_RANK:
        raise RuntimeError("special_chebyshev_polynomial_v: rank > 8 unsupported")
    pad = _MAX_RANK - nd
    sh = [1] * pad + [int(v) for v in out_shape]
    xstr = [0] * pad + _broadcast_strides(x, out_shape)
    nstr = [0] * pad + _broadcast_strides(n, out_shape)
    _chev_v_strided[((total + _BLOCK - 1) // _BLOCK,)](
        x,
        n,
        out,
        total,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sh[6],
        sh[7],
        xstr[0],
        xstr[1],
        xstr[2],
        xstr[3],
        xstr[4],
        xstr[5],
        xstr[6],
        xstr[7],
        nstr[0],
        nstr[1],
        nstr[2],
        nstr[3],
        nstr[4],
        nstr[5],
        nstr[6],
        nstr[7],
        BLOCK=_BLOCK,
        N_SCALAR=(n.numel() == 1),
        FP64=(x.dtype == torch.float64),
        num_warps=_WARPS,
    )
    return out
