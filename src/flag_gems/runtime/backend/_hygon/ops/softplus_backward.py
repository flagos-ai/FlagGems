# Copyright 2026 FlagOS Contributors
# Copyright (c) 2024 FlagGems. All rights reserved.
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
import struct

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

"""Bounded compiled-code cache using public Triton launchers.

Only argument type, alignment and scalar metadata specialize code; tensor data
and returned values are never retained. Current stream and runtime hooks are
resolved by each public CompiledKernel grid launch.
"""
_hygon_launch_cached__CACHE = {}


def _hygon_launch_cached__launch(kernel, grid, *args, **kwargs):
    names = kernel.arg_names
    full_args = args + tuple((kwargs[name] for name in names[len(args) :]))
    options = {name: value for (name, value) in kwargs.items() if name not in names}
    signature = tuple(
        (
            (
                (value.dtype, value.data_ptr() % 16 == 0)
                if isinstance(value, torch.Tensor)
                else value
            )
            for value in full_args
        )
    )
    key = (
        kernel,
        torch.cuda.current_device(),
        signature,
        tuple(sorted(options.items())),
    )
    compiled = _hygon_launch_cached__CACHE.get(key)
    if compiled is None:
        compiled = kernel[grid](*full_args, **options)
        if compiled is not None:
            if len(_hygon_launch_cached__CACHE) >= 256:
                _hygon_launch_cached__CACHE.pop(next(iter(_hygon_launch_cached__CACHE)))
            _hygon_launch_cached__CACHE[key] = compiled
    else:
        dims = tuple(grid) + (1,) * (3 - len(grid))
        compiled[dims](*full_args, stream=torch.cuda.current_stream().cuda_stream)


launch = _hygon_launch_cached__launch
logger = logging.getLogger(__name__)
_DENSE_KERNELS = {}
_NUM_WARPS = 4
_BLOCK_2BYTE = 2048
_BLOCK_OTHER = 1024


def _f64_bits(v):
    return struct.unpack("<q", struct.pack("<d", float(v)))[0]


def _block_for(itemsize):
    return _BLOCK_2BYTE if itemsize <= 2 else _BLOCK_OTHER


@triton.jit
def _softplus_bwd_contig(
    g_ptr,
    x_ptr,
    o_ptr,
    n,
    beta,
    threshold,
    beta_bits,
    thresh_bits,
    BLOCK: tl.constexpr,
    NUMEL_I64: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUMEL_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if F64:
        gv = g.to(tl.float64)
        xv = x.to(tl.float64)
        b = tl.cast(beta_bits, tl.float64, bitcast=True)
        th = tl.cast(thresh_bits, tl.float64, bitcast=True)
    else:
        gv = g.to(tl.float32)
        xv = x.to(tl.float32)
        b = beta
        th = threshold
    t = xv * b
    z = tl_extra_shim.exp(t) if F64 else tl.exp(t)
    y = tl.where(t > th, gv, gv * (z / (1.0 + z)))
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softplus_bwd_strided(
    g_ptr,
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
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    g7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    beta,
    threshold,
    beta_bits,
    thresh_bits,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    q = rem % d7
    rem = rem // d7
    goff = q * g7
    xoff = q * x7
    ooff = q * o7
    q = rem % d6
    rem = rem // d6
    goff += q * g6
    xoff += q * x6
    ooff += q * o6
    q = rem % d5
    rem = rem // d5
    goff += q * g5
    xoff += q * x5
    ooff += q * o5
    q = rem % d4
    rem = rem // d4
    goff += q * g4
    xoff += q * x4
    ooff += q * o4
    q = rem % d3
    rem = rem // d3
    goff += q * g3
    xoff += q * x3
    ooff += q * o3
    q = rem % d2
    rem = rem // d2
    goff += q * g2
    xoff += q * x2
    ooff += q * o2
    q = rem % d1
    rem = rem // d1
    goff += q * g1
    xoff += q * x1
    ooff += q * o1
    q = rem % d0
    goff += q * g0
    xoff += q * x0
    ooff += q * o0
    g = tl.load(g_ptr + goff, mask=mask, other=0.0)
    x = tl.load(x_ptr + xoff, mask=mask, other=0.0)
    if F64:
        gv = g.to(tl.float64)
        xv = x.to(tl.float64)
        b = tl.cast(beta_bits, tl.float64, bitcast=True)
        th = tl.cast(thresh_bits, tl.float64, bitcast=True)
    else:
        gv = g.to(tl.float32)
        xv = x.to(tl.float32)
        b = beta
        th = threshold
    t = xv * b
    z = tl_extra_shim.exp(t) if F64 else tl.exp(t)
    y = tl.where(t > th, gv, gv * (z / (1.0 + z)))
    tl.store(o_ptr + ooff, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softplus_fwd_contig(
    x_ptr,
    o_ptr,
    n,
    beta,
    threshold,
    beta_bits,
    thresh_bits,
    BLOCK: tl.constexpr,
    NUMEL_I64: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUMEL_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if F64:
        xv = x.to(tl.float64)
        b = tl.cast(beta_bits, tl.float64, bitcast=True)
        th = tl.cast(thresh_bits, tl.float64, bitcast=True)
    else:
        xv = x.to(tl.float32)
        b = beta
        th = threshold
    t = xv * b
    y = (
        tl.where(t > th, xv, tl_extra_shim.log1p(tl_extra_shim.exp(t)) / b)
        if F64
        else tl.where(t > th, xv, tl.log(1.0 + tl.exp(t)) / b)
    )
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=mask)


def _pad8(v, fill):
    v = list(v)
    assert len(v) <= 8
    return v + [fill] * (8 - len(v))


def _strides_of(t, out_shape):
    ndim = len(out_shape)
    pad = ndim - t.dim()
    s = [0] * pad + list(t.stride())
    for i in range(pad, ndim):
        if t.shape[i - pad] == 1 and out_shape[i] != 1:
            s[i] = 0
    return s


def _backward_dense(g, x, beta, threshold):
    out = torch.empty_like(g)
    n = out.numel()
    if n == 0:
        return out
    f64 = out.dtype == torch.float64
    block = _block_for(out.dtype.itemsize)
    grid = ((n + block - 1) // block,)
    args = (
        g,
        x,
        out,
        n,
        beta,
        threshold,
        _f64_bits(beta) if f64 else 0,
        _f64_bits(threshold) if f64 else 0,
        block,
        n > 2**31 - 1,
        f64,
    )
    key = (
        g.get_device(),
        g.dtype,
        n,
        beta,
        threshold,
        g.data_ptr() % 16 == 0,
        x.data_ptr() % 16 == 0,
        out.data_ptr() % 16 == 0,
    )
    compiled = _DENSE_KERNELS.get(key)
    if compiled is None:
        compiled = _softplus_bwd_contig[grid](*args, num_warps=_NUM_WARPS)
        if compiled is not None:
            if len(_DENSE_KERNELS) >= 256:
                _DENSE_KERNELS.pop(next(iter(_DENSE_KERNELS)))
            _DENSE_KERNELS[key] = compiled
    else:
        compiled[grid[0], 1, 1](*args, stream=torch.cuda.current_stream().cuda_stream)
    return out


def _backward_broadcast(g, x, beta, threshold):
    out_shape = torch.broadcast_shapes(tuple(g.shape), tuple(x.shape))
    out_dtype = torch.promote_types(g.dtype, x.dtype)
    out = torch.empty(out_shape, dtype=out_dtype, device=g.device)
    n = out.numel()
    if n == 0:
        return out
    f64 = out_dtype == torch.float64
    block = _block_for(out.dtype.itemsize)
    grid = ((n + block - 1) // block,)
    ndim = len(out_shape)
    if ndim > 8:
        raise RuntimeError("softplus_backward: ranks above 8 are not supported")
    if (
        g.shape == out_shape
        and x.shape == out_shape
        and g.is_contiguous()
        and x.is_contiguous()
    ):
        launch(
            _softplus_bwd_contig,
            grid,
            g,
            x,
            out,
            n,
            beta,
            threshold,
            _f64_bits(beta) if f64 else 0,
            _f64_bits(threshold) if f64 else 0,
            BLOCK=block,
            NUMEL_I64=n > 2**31 - 1,
            F64=f64,
            num_warps=_NUM_WARPS,
        )
        return out
    launch(
        _softplus_bwd_strided,
        grid,
        g,
        x,
        out,
        n,
        *_pad8(list(out_shape), 1),
        *_pad8(_strides_of(g, out_shape), 0),
        *_pad8(_strides_of(x, out_shape), 0),
        *_pad8(list(out.stride()), 0),
        beta,
        threshold,
        _f64_bits(beta) if f64 else 0,
        _f64_bits(threshold) if f64 else 0,
        BLOCK=block,
        F64=f64,
        num_warps=_NUM_WARPS,
    )
    return out


def _backward(g, x, beta, threshold):
    if (
        g.dtype == x.dtype
        and g.shape == x.shape
        and g.is_contiguous()
        and x.is_contiguous()
    ):
        return _backward_dense(g, x, beta, threshold)
    return _backward_broadcast(g, x, beta, threshold)


class _SoftplusFunction(torch.autograd.Function):
    """Autograd bridge for direct functional use.

    Both the forward value and the backward value are produced by the Triton
    kernels above; this class only carries the saved tensor and the scalars.
    """

    @staticmethod
    def forward(ctx, x, beta, threshold):
        if not x.is_contiguous():
            raise RuntimeError("softplus forward helper requires a dense input")
        ctx.save_for_backward(x)
        ctx.beta = beta
        ctx.threshold = threshold
        out = torch.empty_like(x)
        n = out.numel()
        if n == 0:
            return out
        f64 = out.dtype == torch.float64
        block = _block_for(out.dtype.itemsize)
        grid = ((n + block - 1) // block,)
        launch(
            _softplus_fwd_contig,
            grid,
            x,
            out,
            n,
            beta,
            threshold,
            _f64_bits(beta) if f64 else 0,
            _f64_bits(threshold) if f64 else 0,
            BLOCK=block,
            NUMEL_I64=n > 2**31 - 1,
            F64=f64,
            num_warps=_NUM_WARPS,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (_backward(grad_output, x, ctx.beta, ctx.threshold), None, None)


def softplus_backward(grad_output, self=None, beta=1.0, threshold=20.0):
    logger.debug("GEMS_HYGON SOFTPLUS_BACKWARD")
    if type(beta) is not float:
        beta = float(beta)
    if type(threshold) is not float:
        threshold = float(threshold)
    if self is None:
        return _SoftplusFunction.apply(grad_output, beta, threshold)
    return _backward(grad_output, self, beta, threshold)
