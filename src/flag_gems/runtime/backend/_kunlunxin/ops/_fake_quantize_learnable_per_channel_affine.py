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

"""Kunlunxin (XPU3) implementation of
``aten::_fake_quantize_learnable_per_channel_affine`` (forward only).

Why an override is needed
-------------------------
The generic implementation (``src/flag_gems/ops/...``) computes the fp32
reciprocal with ``tl_extra_shim.rcp_rn``.  On this backend ``rcp_rn`` has no
device implementation, so the ELF link step fails with
``ld.lld: error: undefined symbol: Unsupported`` for *every* dtype/shape: the
op cannot compile at all.  ``tl.div_rn(1.0f, scale)`` is the backend's
correctly-rounded fp32 reciprocal (same value as C's ``1.0f / scale``, which is
what ATen multiplies by), so it is used instead.

Numerics (bit-identical to the ATen CPU/CUDA reference, verified maxdiff == 0)
------------------------------------------------------------------------------
    z = rne(zero_point)                      # round-half-to-even, then clamp
    z = clamp(z, quant_min, quant_max)       # (ATen rounds before clamping)
    q = rne(x * fp32(1 / scale)) + z
    q = clamp(q, quant_min, quant_max)
    out = (q - z) * scale

``rne`` (round-half-to-even) uses the fp32 magic identity
``(v + 1.5*2**23) - 1.5*2**23``, exact for ``|v| < 2**22``; outside that range
the value is already far outside ``[quant_min, quant_max]`` so the clamp
saturates identically (see
``wiki/techniques/xpu3-magic-number-round-half-even.md``).

Two launch shapes (see ``harness/solution/<op>/README.md``)
-----------------------------------------------------------
* ``_fq_flat_kernel``: one strictly 1-D grid of unmasked blocks plus (only when
  ``numel % BLOCK != 0``) exactly one masked tail program.  The channel index is
  a per-lane expression, which costs a small cached gather of ``scale`` /
  ``zero_point`` -- measured at ~0.37 ns/elem, 3.5x faster than ATen even for the
  worst case (last-dim channels, ``channel_stride == 1``).
* ``_fq_row_kernel``: one *contiguous* run of one channel slice per program
  (``channel_stride >= 512``), scalar ``scale``/``zero_point`` load.  Each slice
  is covered by ``NCHUNK`` blocks whose start is clamped to ``CS - BLOCK``
  (compile-time constants), so the overlapping tail lanes recompute identical
  values and no mask / per-lane clamp is emitted.  Measured 442 us vs 16.7 ms for
  the flat gather on ``(16,128,64,60) axis=1``.

Red lines: no torch/aten compute and none of the forbidden aten copy primitives;
the single ``empty_like`` below is a pure allocation, and ``contiguous()`` (a real
Gems kernel) is only called for a non-contiguous input.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.contiguous import contiguous
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@triton.jit
def _fq_quant(x, s, z, qmin, qmax):
    # fp32 reciprocal == C's ``1.0f / scale``; ATen multiplies, it does not divide.
    inv = tl.div_rn(1.0, s)
    y = x * inv
    # round-half-to-even: (y + 1.5*2**23) - 1.5*2**23, exact for |y| < 2**22.
    r = (y + 12582912.0) - 12582912.0
    q = r + z
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    return (q - z) * s


@triton.jit
def _fq_flat_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    n_elem,
    n_full,
    channel_stride,
    qmin,
    qmax,
    C: tl.constexpr,
    LOG2_CS: tl.constexpr,
    BLOCK: tl.constexpr,
    TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    if TAIL and pid >= n_full:
        # exactly one tail program: per-lane clamp + mask (the vectorizer cannot
        # keep this on the block-DMA path, but it only costs one tile).
        off = n_full * BLOCK + tl.arange(0, BLOCK)
        valid = off < n_elem
        x = tl.where(
            valid, tl.load(x_ptr + tl.minimum(off, n_elem - 1)).to(tl.float32), 0.0
        )
        if LOG2_CS >= 0:
            c = (off >> LOG2_CS) % C
        else:
            c = (off // channel_stride) % C
        s = tl.load(scale_ptr + c).to(tl.float32)
        z = tl.minimum(
            tl.maximum(
                (tl.load(zp_ptr + c).to(tl.float32) + 12582912.0) - 12582912.0, qmin
            ),
            qmax,
        )
        out = _fq_quant(x, s, z, qmin, qmax)
        tl.store(
            out_ptr + tl.minimum(off, n_elem - 1),
            tl.where(valid, out, 0.0).to(out_ptr.dtype.element_ty),
            mask=valid,
        )
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off).to(tl.float32)
        if LOG2_CS >= 0:
            c = (off >> LOG2_CS) % C
        else:
            c = (off // channel_stride) % C
        s = tl.load(scale_ptr + c).to(tl.float32)
        z = tl.minimum(
            tl.maximum(
                (tl.load(zp_ptr + c).to(tl.float32) + 12582912.0) - 12582912.0, qmin
            ),
            qmax,
        )
        tl.store(out_ptr + off, _fq_quant(x, s, z, qmin, qmax).to(out_ptr.dtype.element_ty))


@triton.jit
def _fq_row_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    qmin,
    qmax,
    C: tl.constexpr,
    CS: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    # One channel slice (CS contiguous elements, single channel) per program.
    row = tl.program_id(0)
    c = row % C
    base = row * CS
    s = tl.load(scale_ptr + c).to(tl.float32)
    z = tl.minimum(
        tl.maximum(
            (tl.load(zp_ptr + c).to(tl.float32) + 12582912.0) - 12582912.0, qmin
        ),
        qmax,
    )
    for i in tl.static_range(NCHUNK):
        # constexpr start; the last chunk overlaps backwards instead of masking.
        start = tl.minimum(i * BLOCK, CS - BLOCK)
        off = base + start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off).to(tl.float32)
        tl.store(out_ptr + off, _fq_quant(x, s, z, qmin, qmax).to(out_ptr.dtype.element_ty))


@triton.jit
def _fq_row_range_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    qmin,
    qmax,
    C: tl.constexpr,
    CS: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    # Same as _fq_row_kernel but with a real loop (unrolling > 4 chunks blows
    # the XPU3 local-memory / code budget: "out of resource: uni_sram").
    row = tl.program_id(0)
    c = row % C
    base = row * CS
    s = tl.load(scale_ptr + c).to(tl.float32)
    z = tl.minimum(
        tl.maximum(
            (tl.load(zp_ptr + c).to(tl.float32) + 12582912.0) - 12582912.0, qmin
        ),
        qmax,
    )
    for i in tl.range(NCHUNK):
        start = tl.minimum(i * BLOCK, CS - BLOCK)
        off = base + start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + off).to(tl.float32)
        tl.store(out_ptr + off, _fq_quant(x, s, z, qmin, qmax).to(out_ptr.dtype.element_ty))


def _next_pow2(v):
    return 1 << (v - 1).bit_length()


def _pow2_le(v):
    return 1 << (v.bit_length() - 1)


def _pick_flat_block(n_elem):
    # Measured on P800 xpu3 (harness/perf_ir/probe_fqlpca_small.py):
    # n=8 -> 16 (5.2us), n=32 -> 64 (8.0us), n=266 -> 512 (9.5us); then the
    # tier table keeps the block-DMA run wide (16K lanes for the big shapes).
    if n_elem < 512:
        return max(16, _next_pow2(n_elem))
    if n_elem <= 1024:
        return 512
    if n_elem <= 16384:
        return 2048
    if n_elem <= 131072:
        return 8192
    return 16384


def _launch_flat(x, scale, zp, out, n_elem, channel_stride, n_channels, qmin, qmax):
    block = _pick_flat_block(n_elem)
    is_pow2 = (channel_stride & (channel_stride - 1)) == 0
    log2_cs = channel_stride.bit_length() - 1 if is_pow2 else -1
    n_full = n_elem // block
    tail = n_elem % block != 0
    grid = (n_full + (1 if tail else 0),)
    with torch_device_fn.device(x.device):
        _fq_flat_kernel[grid](
            x,
            scale,
            zp,
            out,
            n_elem,
            n_full,
            channel_stride,
            qmin,
            qmax,
            C=n_channels,
            LOG2_CS=log2_cs,
            BLOCK=block,
            TAIL=tail,
            num_warps=8,
        )


def _launch_row(x, scale, zp, out, channel_stride, n_channels, qmin, qmax):
    block = min(2048, _pow2_le(channel_stride))
    n_chunk = triton.cdiv(channel_stride, block)
    grid = (x.numel() // channel_stride,)
    kernel = _fq_row_kernel if n_chunk <= 4 else _fq_row_range_kernel
    with torch_device_fn.device(x.device):
        kernel[grid](
            x,
            scale,
            zp,
            out,
            qmin,
            qmax,
            C=n_channels,
            CS=channel_stride,
            BLOCK=block,
            NCHUNK=n_chunk,
            num_warps=8,
        )


def _fake_quantize_learnable_per_channel_affine(
    self, scale, zero_point, axis, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE")

    if self.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"unsupported dtype {self.dtype}; only floating-point dtypes are supported"
        )

    ndim = self.dim()
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, "
            f"{ndim - 1}], but got {axis})"
        )

    x = self if self.is_contiguous() else contiguous(self)
    s = scale if scale.is_contiguous() else contiguous(scale)
    z = zero_point if zero_point.is_contiguous() else contiguous(zero_point)

    n_elem = x.numel()
    out = torch.empty_like(x)
    if n_elem == 0:
        # host-side short-circuit: never launch a 0-program grid
        return out

    shape = x.shape
    n_channels = shape[axis]
    channel_stride = 1
    for i in range(axis + 1, ndim):
        channel_stride *= shape[i]

    qmin = float(quant_min)
    qmax = float(quant_max)

    n_slices = n_elem // channel_stride
    if channel_stride >= 512 and n_slices >= 256:
        _launch_row(x, s, z, out, channel_stride, n_channels, qmin, qmax)
    else:
        _launch_flat(x, s, z, out, n_elem, channel_stride, n_channels, qmin, qmax)
    return out
