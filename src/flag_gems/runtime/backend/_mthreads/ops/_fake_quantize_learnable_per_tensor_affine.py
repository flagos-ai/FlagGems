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

"""Triton implementation of _fake_quantize_learnable_per_tensor_affine.

Reference semantics, pinned numerically against torch 2.9.1 on the MUSA
target (see the debug evidence):

    s   = float32(scale)                            # single-element tensor
    zp  = round_half_even(float32(zero_point))      # fractional zp -> nearest int
    inv = 1.0f / s                                  # correctly rounded
    p   = round_half_even(x * inv)                  # fp32 opmath, all dtypes
    q   = clamp(p + zp, quant_min, quant_max)       # zp added AFTER rounding
    y   = (q - zp) * s                              # fp32
    out = cast(y, dtype(x))

Adding the integer zero_point after the rounding (rather than rounding
`x*inv + zp`) is what the aten kernel does, and it changes every exact tie:
`round_half_even(89.5) + 127 = 89` while `round_half_even(89.5 + 127) = 90`.
With scale=0.1 and hundreds of millions of elements such ties are common, so
this ordering is mandatory.  Rounding is done with the ±1.5*2**23 magic
constant (exact ties-to-even for |v| < 2**22; out-of-range magnitudes are
clamped away afterwards, and +/-inf passes through to qmax/qmin).  A NaN input
folds to q=0, matching the device reference.  grad_factor is backward-only.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._fake_quantize_learnable_per_tensor_affine import (
    _fake_quantize_learnable_per_tensor_affine as default_fq_learnable,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so fp64 falls back to the
# generic implementation; fp16/bf16/fp32 are specialized.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_MAGIC = 12582912.0  # 1.5 * 2**23

_BLOCK = 4096
_NUM_WARPS = 2

# Latency-bound shapes: a single 4096-element block leaves most SMs idle and
# gives every thread a 16-element dependent chain, and each extra runtime
# argument costs real Python dispatch time in a ~3 us launch.  Small tensors
# therefore get a dedicated entry point with four runtime arguments and a
# 1024-element block over 4 warps.
_SMALL_N = 16384
_SMALL_BLOCK = 1024
_SMALL_WARPS = 4


@triton.jit
def _fq_small(
    x_ptr,
    s_ptr,
    zp_ptr,
    out_ptr,
    QUANT_MIN: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Latency-bound path: four runtime arguments, unmasked access (the
    dispatch requires numel % BLOCK == 0)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    s = tl.load(s_ptr).to(tl.float32)
    zp = tl.load(zp_ptr).to(tl.float32)
    zp = (zp + 12582912.0) - 12582912.0
    qmin = 0.0 + QUANT_MIN
    qmax = 0.0 + QUANT_MAX
    zp = tl.minimum(tl.maximum(zp, qmin), qmax)
    inv = tl.math.div_rn(1.0, s)

    x = tl.load(x_ptr + offs).to(tl.float32)
    p = x * inv
    p = (p + 12582912.0) - 12582912.0
    q = p + zp
    q = tl.where(x != x, 0.0, q)
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    y = (q - zp) * s

    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["n"])
def _fq_kernel(
    x_ptr,
    s_ptr,
    zp_ptr,
    out_ptr,
    n,
    quant_min,
    quant_max,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # --- per-tensor quantization parameters (uniform scalar loads) ----------
    s = tl.load(s_ptr).to(tl.float32)
    zp = tl.load(zp_ptr).to(tl.float32)
    zp = (zp + 12582912.0) - 12582912.0  # round-half-to-even
    qmin = quant_min.to(tl.float32)
    qmax = quant_max.to(tl.float32)
    zp = tl.minimum(tl.maximum(zp, qmin), qmax)
    inv = tl.math.div_rn(1.0, s)

    if EVEN:
        # numel is a multiple of BLOCK: no predicate, so the backend can emit
        # full-width vector loads/stores on the streaming path.  Both accesses
        # are marked evict-first: the tensors are streamed once, so keeping
        # them out of the cache hierarchy leaves L2 to absorb the store bursts.
        x = tl.load(x_ptr + offs, eviction_policy="evict_first").to(tl.float32)
        p = x * inv
        p = (p + 12582912.0) - 12582912.0  # round-half-to-even of the product
        q = p + zp
        q = tl.where(x != x, 0.0, q)  # NaN input -> q = 0 (device behaviour)
        q = tl.minimum(tl.maximum(q, qmin), qmax)
        y = (q - zp) * s
        tl.store(
            out_ptr + offs,
            y.to(out_ptr.dtype.element_ty),
            eviction_policy="evict_first",
        )
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = x * inv
        p = (p + 12582912.0) - 12582912.0
        q = p + zp
        q = tl.where(x != x, 0.0, q)
        q = tl.minimum(tl.maximum(q, qmin), qmax)
        y = (q - zp) * s
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _fq_kernel_strided(
    x_ptr,
    s_ptr,
    zp_ptr,
    out_ptr,
    n,
    quant_min,
    quant_max,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    st6,
    st7,
    sz0,
    sz1,
    sz2,
    sz3,
    sz4,
    sz5,
    sz6,
    sz7,
    BLOCK: tl.constexpr,
):
    """General (non-contiguous) path: logical row-major index -> strided
    address.  Shape/stride lists are left-padded with (size 1, stride 0), so a
    single specialization serves every rank up to 8."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    s = tl.load(s_ptr).to(tl.float32)
    zp = tl.load(zp_ptr).to(tl.float32)
    zp = (zp + 12582912.0) - 12582912.0
    qmin = quant_min.to(tl.float32)
    qmax = quant_max.to(tl.float32)
    zp = tl.minimum(tl.maximum(zp, qmin), qmax)
    inv = tl.math.div_rn(1.0, s)

    idx = offs.to(tl.int64)
    i7 = idx % sz7
    idx = idx // sz7
    i6 = idx % sz6
    idx = idx // sz6
    i5 = idx % sz5
    idx = idx // sz5
    i4 = idx % sz4
    idx = idx // sz4
    i3 = idx % sz3
    idx = idx // sz3
    i2 = idx % sz2
    idx = idx // sz2
    i1 = idx % sz1
    i0 = idx // sz1
    addr = (
        i0 * st0
        + i1 * st1
        + i2 * st2
        + i3 * st3
        + i4 * st4
        + i5 * st5
        + i6 * st6
        + i7 * st7
    )

    x = tl.load(x_ptr + addr, mask=mask, other=0.0).to(tl.float32)
    p = (x * inv + 12582912.0) - 12582912.0
    q = p + zp
    q = tl.where(x != x, 0.0, q)
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    y = (q - zp) * s

    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _pad8(values, fill):
    return [fill] * (8 - len(values)) + list(values)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _fake_quantize_learnable_per_tensor_affine(
    self, scale, zero_point, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS_MTHREADS _FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE")

    x = self
    if not _use_triton_kernel(x):
        return default_fq_learnable(
            x, scale, zero_point, quant_min, quant_max, grad_factor
        )

    n = x.numel()
    qmin = int(quant_min)
    qmax = int(quant_max)

    with torch_device_fn.device(x.device):
        if x.is_contiguous():
            out = torch.empty_like(x)
            if n == 0:
                return out
            if n <= _SMALL_N and n % _SMALL_BLOCK == 0:
                _fq_small[(n // _SMALL_BLOCK,)](
                    x,
                    scale,
                    zero_point,
                    out,
                    QUANT_MIN=qmin,
                    QUANT_MAX=qmax,
                    BLOCK=_SMALL_BLOCK,
                    num_warps=_SMALL_WARPS,
                )
                return out
            if x.dim() == 0:
                xf = x.unsqueeze(0)
                of = out.unsqueeze(0)
            else:
                xf = x.view(-1)
                of = out.view(-1)
            grid = (triton.cdiv(n, _BLOCK),)
            _fq_kernel[grid](
                xf,
                scale,
                zero_point,
                of,
                n,
                qmin,
                qmax,
                EVEN=(n % _BLOCK) == 0,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
            return out

        # Non-contiguous source: never exercised by the workloads, kept correct.
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if n == 0:
            return out
        grid = (triton.cdiv(n, _BLOCK),)
        _fq_kernel_strided[grid](
            x,
            scale,
            zero_point,
            out,
            n,
            qmin,
            qmax,
            *_pad8(x.stride(), 0),
            *_pad8(x.shape, 1),
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return out
