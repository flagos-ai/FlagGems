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
"""Triton implementation of ``aten::_foreach_reciprocal``.

For every tensor ``t`` in the input sequence: ``out = 1 / t`` elementwise,
returned as a freshly allocated tensor in input order, leaving ``t`` untouched.

Semantics reproduced from the ATen reference:
  * float16/bfloat16/float32 -> computed in fp32 and rounded back (RTNE)
  * float64                  -> computed in fp64
  * complex64/complex128     -> (a - bi) / (a^2 + b^2) via zero-copy views
  * integral/bool            -> promoted to float32
  * empty tensor list        -> RuntimeError
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import (
    _foreach_reciprocal as default__foreach_reciprocal,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_BLOCK = 1024
_WARPS = 4
_MAXD = 8
_INT64_LIMIT = 2147483647

_FP16 = torch.float16
_BF16 = torch.bfloat16
_FP32 = torch.float32
_FP64 = torch.float64
_FAST_DTYPES = (_FP16, _BF16, _FP32)


@triton.jit
def _recip_flat_kernel(
    x_ptr, y_ptr, n, COMPLEX: tl.constexpr, FP64: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if COMPLEX:
        i = offs.to(tl.int64) * 2
        xr = tl.load(x_ptr + i, mask=mask, other=1.0)
        xi = tl.load(x_ptr + i + 1, mask=mask, other=0.0)
        d = xr * xr + xi * xi
        tl.store(y_ptr + i, xr / d, mask=mask)
        tl.store(y_ptr + i + 1, -xi / d, mask=mask)
    else:
        x = tl.load(x_ptr + offs, mask=mask, other=1)
        if FP64:
            tl.store(y_ptr + offs, 1.0 / x, mask=mask)
        else:
            tl.store(y_ptr + offs, 1.0 / x.to(tl.float32), mask=mask)


@triton.jit
def _recip_flat_long_kernel(
    x_ptr, y_ptr, n, COMPLEX: tl.constexpr, FP64: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    if COMPLEX:
        i = offs * 2
        xr = tl.load(x_ptr + i, mask=mask, other=1.0)
        xi = tl.load(x_ptr + i + 1, mask=mask, other=0.0)
        d = xr * xr + xi * xi
        tl.store(y_ptr + i, xr / d, mask=mask)
        tl.store(y_ptr + i + 1, -xi / d, mask=mask)
    else:
        x = tl.load(x_ptr + offs, mask=mask, other=1)
        if FP64:
            tl.store(y_ptr + offs, 1.0 / x, mask=mask)
        else:
            tl.store(y_ptr + offs, 1.0 / x.to(tl.float32), mask=mask)


@triton.jit
def _recip_stride_kernel(
    x_ptr,
    y_ptr,
    n,
    xs,
    ys,
    COMPLEX: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    if COMPLEX:
        i = offs * xs * 2
        j = offs * ys * 2
        xr = tl.load(x_ptr + i, mask=mask, other=1.0)
        xi = tl.load(x_ptr + i + 1, mask=mask, other=0.0)
        d = xr * xr + xi * xi
        tl.store(y_ptr + j, xr / d, mask=mask)
        tl.store(y_ptr + j + 1, -xi / d, mask=mask)
    else:
        x = tl.load(x_ptr + offs * xs, mask=mask, other=1)
        if FP64:
            tl.store(y_ptr + offs * ys, 1.0 / x, mask=mask)
        else:
            tl.store(y_ptr + offs * ys, 1.0 / x.to(tl.float32), mask=mask)


@triton.jit
def _recip_nd_kernel(
    x_ptr,
    y_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    ys0,
    ys1,
    ys2,
    ys3,
    ys4,
    ys5,
    ys6,
    ys7,
    COMPLEX: tl.constexpr,
    FP64: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rem = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = rem < n
    xo = tl.zeros([BLOCK], dtype=tl.int64)
    yo = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM > 7:
        i = rem % s7
        rem = rem // s7
        xo += i * xs7
        yo += i * ys7
    if NDIM > 6:
        i = rem % s6
        rem = rem // s6
        xo += i * xs6
        yo += i * ys6
    if NDIM > 5:
        i = rem % s5
        rem = rem // s5
        xo += i * xs5
        yo += i * ys5
    if NDIM > 4:
        i = rem % s4
        rem = rem // s4
        xo += i * xs4
        yo += i * ys4
    if NDIM > 3:
        i = rem % s3
        rem = rem // s3
        xo += i * xs3
        yo += i * ys3
    if NDIM > 2:
        i = rem % s2
        rem = rem // s2
        xo += i * xs2
        yo += i * ys2
    if NDIM > 1:
        i = rem % s1
        rem = rem // s1
        xo += i * xs1
        yo += i * ys1
    if NDIM > 0:
        xo += rem * xs0
        yo += rem * ys0
    if COMPLEX:
        xo = xo * 2
        yo = yo * 2
        xr = tl.load(x_ptr + xo, mask=mask, other=1.0)
        xi = tl.load(x_ptr + xo + 1, mask=mask, other=0.0)
        d = xr * xr + xi * xi
        tl.store(y_ptr + yo, xr / d, mask=mask)
        tl.store(y_ptr + yo + 1, -xi / d, mask=mask)
    else:
        x = tl.load(x_ptr + xo, mask=mask, other=1)
        if FP64:
            tl.store(y_ptr + yo, 1.0 / x, mask=mask)
        else:
            tl.store(y_ptr + yo, 1.0 / x.to(tl.float32), mask=mask)


def _launch_strided(src, dst, n, complex_flag, fp64):
    if n is None:
        n = (src.numel() >> 1) if complex_flag else src.numel()
    shape = src.shape
    nd = len(shape)
    if nd > _MAXD:
        for i in range(shape[0]):
            _launch_strided(
                src.select(0, i), dst.select(0, i), None, complex_flag, fp64
            )
        return
    if nd <= 1:
        if nd == 0:
            grid = ((n + _BLOCK - 1) // _BLOCK,)
            _recip_flat_long_kernel[grid](
                src,
                dst,
                n,
                COMPLEX=complex_flag,
                FP64=fp64,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
            return
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _recip_stride_kernel[grid](
            src,
            dst,
            n,
            src.stride(0),
            dst.stride(0),
            COMPLEX=complex_flag,
            FP64=fp64,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
        return
    pad = _MAXD - nd
    shp = tuple(shape) + (1,) * pad
    xs = tuple(src.stride()) + (0,) * pad
    ys = tuple(dst.stride()) + (0,) * pad
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _recip_nd_kernel[grid](
        src,
        dst,
        n,
        *shp,
        *xs,
        *ys,
        COMPLEX=complex_flag,
        FP64=fp64,
        NDIM=nd,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _launch_into(src, dst, n, complex_flag, fp64):
    """src/dst are float/int storage views; ``n`` is the logical element count."""
    if n == 0:
        return
    if src.is_contiguous():
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        if complex_flag or n > _INT64_LIMIT:
            _recip_flat_long_kernel[grid](
                src,
                dst,
                n,
                COMPLEX=complex_flag,
                FP64=fp64,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
        else:
            _recip_flat_kernel[grid](
                src,
                dst,
                n,
                COMPLEX=False,
                FP64=fp64,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
        return
    _launch_strided(src, dst, n, complex_flag, fp64)


@triton.jit
def _recip_fused16_kernel(
    x0,
    y0,
    x1,
    y1,
    x2,
    y2,
    x3,
    y3,
    x4,
    y4,
    x5,
    y5,
    x6,
    y6,
    x7,
    y7,
    x8,
    y8,
    x9,
    y9,
    x10,
    y10,
    x11,
    y11,
    x12,
    y12,
    x13,
    y13,
    x14,
    y14,
    x15,
    y15,
    n,
    bpt,
    BLOCK: tl.constexpr,
):
    """One launch for a whole list of equally sized contiguous tensors.

    Program ``pid`` handles block ``pid % bpt`` of tensor ``pid // bpt``.
    Slots at or above the real list length are never selected.
    """
    pid = tl.program_id(0)
    t = pid // bpt
    offs = (pid - t * bpt) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if t == 0:
        x = tl.load(x0 + offs, mask=mask, other=1)
        tl.store(y0 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 1:
        x = tl.load(x1 + offs, mask=mask, other=1)
        tl.store(y1 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 2:
        x = tl.load(x2 + offs, mask=mask, other=1)
        tl.store(y2 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 3:
        x = tl.load(x3 + offs, mask=mask, other=1)
        tl.store(y3 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 4:
        x = tl.load(x4 + offs, mask=mask, other=1)
        tl.store(y4 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 5:
        x = tl.load(x5 + offs, mask=mask, other=1)
        tl.store(y5 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 6:
        x = tl.load(x6 + offs, mask=mask, other=1)
        tl.store(y6 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 7:
        x = tl.load(x7 + offs, mask=mask, other=1)
        tl.store(y7 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 8:
        x = tl.load(x8 + offs, mask=mask, other=1)
        tl.store(y8 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 9:
        x = tl.load(x9 + offs, mask=mask, other=1)
        tl.store(y9 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 10:
        x = tl.load(x10 + offs, mask=mask, other=1)
        tl.store(y10 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 11:
        x = tl.load(x11 + offs, mask=mask, other=1)
        tl.store(y11 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 12:
        x = tl.load(x12 + offs, mask=mask, other=1)
        tl.store(y12 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 13:
        x = tl.load(x13 + offs, mask=mask, other=1)
        tl.store(y13 + offs, 1.0 / x.to(tl.float32), mask=mask)
    elif t == 14:
        x = tl.load(x14 + offs, mask=mask, other=1)
        tl.store(y14 + offs, 1.0 / x.to(tl.float32), mask=mask)
    else:
        x = tl.load(x15 + offs, mask=mask, other=1)
        tl.store(y15 + offs, 1.0 / x.to(tl.float32), mask=mask)


def _try_fused(tensors):
    """Fuse a short list of equally sized contiguous float tensors into one launch."""
    nt = len(tensors)
    if nt < 2 or nt > 16:
        return None
    t0 = tensors[0]
    dt = t0.dtype
    if dt is not _FP32 and dt is not _FP16 and dt is not _BF16:
        return None
    n = t0.numel()
    if n == 0 or n > _INT64_LIMIT or not t0.is_contiguous():
        return None
    for t in tensors:
        if t.dtype is not dt or t.numel() != n or not t.is_contiguous():
            return None
    empty_like = torch.empty_like
    outs = [empty_like(t) for t in tensors]
    last = nt - 1
    args = []
    for i in range(16):
        j = i if i < nt else last
        args.append(tensors[j])
        args.append(outs[j])
    bpt = (n + _BLOCK - 1) // _BLOCK
    _recip_fused16_kernel[(bpt * nt,)](
        *args,
        n,
        bpt,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )
    return outs


def _foreach_reciprocal(*args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_RECIPROCAL")
    _seq = args[0] if args else next(iter(kwargs.values()))
    if not isinstance(_seq, (list, tuple)):
        _seq = [_seq]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_reciprocal(*args, **kwargs)
    if args:
        tensors = args[0]
    elif "self" in kwargs:
        tensors = kwargs["self"]
    else:
        tensors = next(iter(kwargs.values()))
    if not isinstance(tensors, (list, tuple)):
        tensors = list(tensors)
    if not len(tensors):
        raise RuntimeError("Tensor list must have at least one tensor.")

    fused = _try_fused(tensors)
    if fused is not None:
        return fused

    out = []
    empty_like = torch.empty_like
    for t in tensors:
        dt = t.dtype
        if dt is _FP32 or dt is _FP16 or dt is _BF16:
            o = empty_like(t)
            n = t.numel()
            if n:
                if t.is_contiguous():
                    grid = ((n + _BLOCK - 1) // _BLOCK,)
                    if n > _INT64_LIMIT:
                        _recip_flat_long_kernel[grid](
                            t,
                            o,
                            n,
                            COMPLEX=False,
                            FP64=False,
                            BLOCK=_BLOCK,
                            num_warps=_WARPS,
                        )
                    else:
                        _recip_flat_kernel[grid](
                            t,
                            o,
                            n,
                            COMPLEX=False,
                            FP64=False,
                            BLOCK=_BLOCK,
                            num_warps=_WARPS,
                        )
                else:
                    _launch_strided(t, o, n, False, False)
            out.append(o)
        elif dt is _FP64:
            o = empty_like(t)
            n = t.numel()
            _launch_into(t, o, n, False, True)
            out.append(o)
        elif t.is_complex():
            if dt is not torch.complex64 and dt is not torch.complex128:
                raise RuntimeError("unsupported complex dtype " + str(dt))
            o = empty_like(t)
            n = t.numel()
            _launch_into(torch.view_as_real(t), torch.view_as_real(o), n, True, False)
            out.append(o)
        else:
            # integral / bool: ATen promotes to float32
            o = empty_like(t, dtype=_FP32)
            n = t.numel()
            _launch_into(t, o, n, False, False)
            out.append(o)
    return out
