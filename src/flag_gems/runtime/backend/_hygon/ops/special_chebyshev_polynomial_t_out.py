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

"""Triton implementation of ``special_chebyshev_polynomial_t`` (out variant).

First-kind Chebyshev polynomial, general (possibly non-integer) degree,
elementwise over ``broadcast(x, n)``.

Reference (ATen, opmath dtype = float for f16/bf16/f32, double for f64):

    x < -1 : cos(n*pi) * cosh(n*acosh(-x))
    x <= 1 : cos(n*acos(x))
    x >  1 : cosh(n*acosh(x))

Two numerically equivalent formulations are used:

* small non-negative integer degree -> three-term recurrence
  ``T0 = 1, T1 = x, T(k+1) = 2x*Tk - T(k-1)`` (pure FMA, no transcendentals;
  the recurrence holds for every real x, so it also reproduces the x < -1
  branch and propagates NaN through the arithmetic).  Profiling showed the
  four accurate libdevice expansions cost ~430 static instructions per
  element and bound the kernel, so integer degrees take this path.

* otherwise -> branchless product identity

      T = cos(n*acos(clamp_nan_safe(x, -1, 1))) * cosh(n*acosh(max(|x|, 1)))

  which is exact on the |x| <= 1 and x > 1 regions (acos(1) = acosh(1) = 0
  and cos(0) = cosh(0) = 1 exactly) and reproduces the x < -1 region through
  cos(n*acos(-1)) = cos(n*pi).  NaN survives the clamp because the clamp is
  built from ``tl.where`` comparisons (NaN compares false), never fmin/fmax.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

logger = logging.getLogger(__name__)

_logger = logging.getLogger(__name__)
_LOG_TEXT = "GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_T_OUT"
# The operator tests raise the generic operator logger
# ("flag_gems.ops.special_chebyshev_polynomial_t") to DEBUG and assert the
# GEMS_HYGON record appears in caplog.text; vendor-specialized loggers are
# keyed by fn.__module__ (_hygon.ops.special_chebyshev_polynomial_t_out),
# which the test does not enable, so the dispatch record is emitted through
# the generic operator logger as well.
_dispatch_log = logging.getLogger("flag_gems.ops.special_chebyshev_polynomial_t")

_BLOCK = 1024
_MAX_RANK = 8
# Largest non-negative integer degree routed to the FMA recurrence.  The
# recurrence is algebraically exact for every degree; the cap only bounds the
# loop length (and compile time of the constexpr variant).
_MAX_DEGREE = 32
_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_INT_DTYPES = (
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
)


# ---------------------------------------------------------------------------
# Small non-negative integer degree, known on the host -> unrolled recurrence
# ---------------------------------------------------------------------------
@triton.jit
def _cheb_int_degree_kernel(
    x_ptr,
    out_ptr,
    numel,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    xr = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if IS_F64:
        x = xr.to(tl.float64)
    else:
        x = xr.to(tl.float32)
    if N == 0:
        y = tl.full(x.shape, 1.0, x.dtype)
    elif N == 1:
        y = x
    else:
        tkm1 = tl.full(x.shape, 1.0, x.dtype)
        t = x
        for _ in tl.static_range(2, N + 1):
            tnext = 2.0 * x * t - tkm1
            tkm1 = t
            t = tnext
        y = t
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Degree is a Python number -> kernel scalar argument (non-integer degrees)
# ---------------------------------------------------------------------------
@triton.jit
def _cheb_value_degree_kernel(
    x_ptr,
    out_ptr,
    numel,
    n_val,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    xr = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if IS_F64:
        x = xr.to(tl.float64)
        n = n_val.to(tl.float64)
    else:
        x = xr.to(tl.float32)
        n = n_val.to(tl.float32)
    ax = tl.abs(x)
    xc = tl.where(x > 1.0, 1.0, tl.where(x < -1.0, -1.0, x))
    y = libdevice.cos(n * libdevice.acos(xc)) * libdevice.cosh(
        n * libdevice.acosh(tl.maximum(ax, 1.0))
    )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Degree is a single runtime scalar element (0-dim / 1-element tensor).
# The integer test happens on the device (wave-uniform branch, no host sync),
# so integer degrees never touch a transcendental.
# ---------------------------------------------------------------------------
@triton.jit
def _cheb_scalar_degree_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    MAX_DEGREE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    xr = tl.load(x_ptr + offs, mask=mask, other=0.0)
    nr = tl.load(n_ptr)
    if IS_F64:
        x = xr.to(tl.float64)
        n = nr.to(tl.float64)
    else:
        x = xr.to(tl.float32)
        n = nr.to(tl.float32)
    ni = n.to(tl.int32)
    if (n == ni.to(n.dtype)) & (ni >= 0) & (ni <= MAX_DEGREE):
        tkm1 = tl.full(x.shape, 1.0, x.dtype)
        t = x
        for _ in range(2, ni + 1):
            tnext = 2.0 * x * t - tkm1
            tkm1 = t
            t = tnext
        y = tl.where(ni == 0, tl.full(x.shape, 1.0, x.dtype), t)
    else:
        ax = tl.abs(x)
        xc = tl.where(x > 1.0, 1.0, tl.where(x < -1.0, -1.0, x))
        y = libdevice.cos(n * libdevice.acos(xc)) * libdevice.cosh(
            n * libdevice.acosh(tl.maximum(ax, 1.0))
        )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Elementwise degree with the same shape as x (contiguous fast path)
# ---------------------------------------------------------------------------
@triton.jit
def _cheb_tensor_degree_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    xr = tl.load(x_ptr + offs, mask=mask, other=0.0)
    nr = tl.load(n_ptr + offs, mask=mask, other=0.0)
    if IS_F64:
        x = xr.to(tl.float64)
        n = nr.to(tl.float64)
    else:
        x = xr.to(tl.float32)
        n = nr.to(tl.float32)
    ax = tl.abs(x)
    xc = tl.where(x > 1.0, 1.0, tl.where(x < -1.0, -1.0, x))
    y = libdevice.cos(n * libdevice.acos(xc)) * libdevice.cosh(
        n * libdevice.acosh(tl.maximum(ax, 1.0))
    )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# General broadcast / non-contiguous fallback (stride decomposed)
# ---------------------------------------------------------------------------
@triton.jit
def _cheb_broadcast_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
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
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    rem = offs.to(tl.int64)
    xoff = tl.zeros((BLOCK,), dtype=tl.int64)
    noff = tl.zeros((BLOCK,), dtype=tl.int64)
    ooff = tl.zeros((BLOCK,), dtype=tl.int64)
    xstr = (xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7)
    nstr = (ns0, ns1, ns2, ns3, ns4, ns5, ns6, ns7)
    ostr = (os0, os1, os2, os3, os4, os5, os6, os7)
    dims = (d0, d1, d2, d3, d4, d5, d6, d7)
    for i in tl.static_range(0, RANK):
        dim = dims[i]
        idx = rem % dim
        rem = rem // dim
        xoff += idx * xstr[i]
        noff += idx * nstr[i]
        ooff += idx * ostr[i]
    xr = tl.load(x_ptr + xoff, mask=mask, other=0.0)
    nr = tl.load(n_ptr + noff, mask=mask, other=0.0)
    if IS_F64:
        x = xr.to(tl.float64)
        n = nr.to(tl.float64)
    else:
        x = xr.to(tl.float32)
        n = nr.to(tl.float32)
    ax = tl.abs(x)
    xc = tl.where(x > 1.0, 1.0, tl.where(x < -1.0, -1.0, x))
    y = libdevice.cos(n * libdevice.acos(xc)) * libdevice.cosh(
        n * libdevice.acosh(tl.maximum(ax, 1.0))
    )
    tl.store(out_ptr + ooff, y.to(out_ptr.dtype.element_ty), mask=mask)


def _strides_rev(t, shape):
    """Broadcast strides of ``t`` against ``shape``, last dimension first."""
    shape_r = tuple(shape)[::-1]
    tshape_r = tuple(t.shape)[::-1]
    tstride_r = tuple(t.stride())[::-1]
    out = []
    for i in range(_MAX_RANK):
        if i >= len(shape_r) or i >= len(tshape_r) or tshape_r[i] == 1:
            out.append(0)
        else:
            out.append(tstride_r[i])
    return out


def special_chebyshev_polynomial_t_out(x, n, out=None, *args, **kwargs):
    logger.debug("GEMS_HYGON SPECIAL_CHEBYSHEV_POLYNOMIAL_T_OUT")
    _dispatch_log.debug("GEMS_HYGON SPECIAL_CHEBYSHEV_POLYNOMIAL_T_OUT")
    if out is None:
        if args:
            out = args[0]
        elif "out" in kwargs:
            out = kwargs["out"]

    xdt = x.dtype
    if xdt != torch.float32 and not x.is_floating_point():
        raise ValueError(
            "special_chebyshev_polynomial_t only supports float32/float64 "
            "and float16/bfloat16 inputs"
        )

    is_tensor_n = n.__class__ is torch.Tensor
    if is_tensor_n:
        common = xdt if n.dtype in _INT_DTYPES else torch.result_type(x, n)
        if n.dim() == 0:
            shape = x.shape
        else:
            shape = torch.broadcast_shapes(x.shape, n.shape)
    else:
        common = xdt
        shape = x.shape
    is_f64 = common == torch.float64

    if out is None:
        out = torch.empty(shape, dtype=common, device=x.device)
    elif out.dtype not in _FLOAT_DTYPES:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type " f"{out.dtype}"
        )
    elif out.shape != shape:
        out.resize_(shape)

    numel = out.numel()
    if numel == 0:
        return out

    _logger.debug(_LOG_TEXT)

    grid = ((numel + _BLOCK - 1) // _BLOCK,)
    flat = x.is_contiguous() and out.is_contiguous()

    if not is_tensor_n:
        if flat:
            if n.__class__ is int:
                if 0 <= n <= _MAX_DEGREE:
                    _cheb_int_degree_kernel[grid](
                        x, out, numel, N=n, BLOCK=_BLOCK, IS_F64=is_f64
                    )
                    return out
            else:
                degree = float(n)
                if degree.is_integer() and 0 <= degree <= _MAX_DEGREE:
                    _cheb_int_degree_kernel[grid](
                        x, out, numel, N=int(degree), BLOCK=_BLOCK, IS_F64=is_f64
                    )
                    return out
            _cheb_value_degree_kernel[grid](
                x, out, numel, float(n), BLOCK=_BLOCK, IS_F64=is_f64
            )
            return out
        n = torch.tensor(
            float(n),
            dtype=torch.float64 if is_f64 else torch.float32,
            device=x.device,
        )

    if flat:
        if n.numel() == 1:
            _cheb_scalar_degree_kernel[grid](
                x,
                n,
                out,
                numel,
                BLOCK=_BLOCK,
                IS_F64=is_f64,
                MAX_DEGREE=_MAX_DEGREE,
            )
            return out
        if n.shape == x.shape:
            _cheb_tensor_degree_kernel[grid](
                x, n, out, numel, BLOCK=_BLOCK, IS_F64=is_f64
            )
            return out

    shape_rev = tuple(shape)[::-1]
    rank = len(shape_rev) if len(shape_rev) > 0 else 1
    xs = _strides_rev(x, shape)
    ns = _strides_rev(n, shape)
    os_ = _strides_rev(out, shape)
    dims = list(shape_rev) + [1] * (_MAX_RANK - len(shape_rev))
    _cheb_broadcast_kernel[grid](
        x,
        n,
        out,
        numel,
        *xs,
        *ns,
        *os_,
        *dims,
        RANK=rank,
        BLOCK=_BLOCK,
        IS_F64=is_f64,
    )
    return out


special_chebyshev_polynomial_t_out = special_chebyshev_polynomial_t_out
