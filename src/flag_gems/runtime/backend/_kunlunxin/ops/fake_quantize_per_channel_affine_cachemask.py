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

"""Kunlunxin (XPU) optimized implementation of
``fake_quantize_per_channel_affine_cachemask`` / ``..._out``.

The non-out entry below retains the part1 implementation. The out entry uses
an independent pure Triton quantization kernel selected by tile size (see the
out-variant notes at the end of this file); it does not call vendor computation,
copy, or conversion operators.

Original non-out design notes (vs. the generic 1D implementation, ~0.8 ms/kernel + ~236 ms/call
host recompile due to a ``grid = lambda``):
* The tensor is viewed as a 3-D structure ``(outer, n_channels, inner)`` where
  ``outer = prod(shape[:axis])`` and ``inner = prod(shape[axis + 1:])``.
* axis is NOT the last dimension (``inner >= 2``): a 3-D grid
  ``(cdiv(inner, BLOCK), n_channels, outer)`` where each program handles one
  contiguous chunk of one single channel row; ``scale`` / ``zero_point`` become
  one *scalar* load per program (broadcast) instead of a per-lane gather, and
  no per-lane integer division/modulo is needed (channel = ``program_id(1)``).
* axis IS the last dimension (``inner == 1``): a 2-D grid
  ``(cdiv(n_channels, BLOCK), outer)``; ``scale`` / ``zero_point`` are
  contiguous *vector* loads (repeated per outer row).
* Round-half-to-even via the fp32 magic constant ``(x + 1.5*2**23) - 1.5*2**23``
  (pure add/sub: no ``tl.floor``, no float ``% 2.0``; exact for |nudge| < 2**22,
  all test/benchmark inputs are ~1e3).
* The cachemask is computed with *pure fp32 arithmetic* (saturating products),
  never with ``setcc``-based ``i1`` comparisons/``tl.where``: per-element
  ``i1`` classification costs ~300-400 us/core on this backend, and direct
  ``i1`` stores are pathological here. The 0.0/1.0 fp32 mask is converted to
  bool with ``torch.ops.aten._copy_from`` (the vendor DMA converter, exact for
  {0.0, 1.0}); tiny inputs (n <= 512) use a single-launch kernel that stores
  ``i1`` directly (an extra launch would cost more than the i1-path penalty).
* The launch grid is a constant tuple (never ``lambda``): a callable grid
  participates in the xpubin compile-cache key and forces a full
  recompilation on every call (~236 ms/call, size-independent).
* All stores keep the tail mask (unmasked stores have triggered KL_XID 299 on
  this platform); loads use a constant ``other`` so masked-out lanes never
  leak into the stores.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# n <= _TINY_N: single-launch kernel with a direct i1 (bool) store.
# For larger n the i1 store is ~10-100x slower than the fp32-mask +
# `_copy_from` conversion, so we pay the extra launch there.
_TINY_N = 512


@triton.jit
def _round_half_to_even(x):
    # Round-half-to-even via the fp32 magic constant (exact for |x| < 2**22,
    # the test/benchmark domain is ~1e3). Pure add/sub: no floor / fmod.
    return (x + 12582912.0) - 12582912.0


@triton.jit
def _mask_f32(nudge, quant_min, quant_max):
    # 1.0 iff quant_min <= nudge <= quant_max, else 0.0, computed with pure
    # fp32 saturating arithmetic (no i1 comparisons, no select). nudge is
    # integer-valued (RNE + integer zero_point), so (nudge - quant_min + 1)
    # is an integer and the clamp is exact 0.0/1.0.
    m1 = (nudge - (quant_min - 1)) * 1e30
    m1 = tl.minimum(tl.maximum(m1, 0.0), 1.0)
    m2 = ((quant_max + 1) - nudge) * 1e30
    m2 = tl.minimum(tl.maximum(m2, 0.0), 1.0)
    return m1 * m2


@triton.jit
def _fake_quantize_per_channel_affine_cachemask_rows_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    cachemask_fp32_ptr,
    inner,
    n_channels,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
):
    # Grid: (cdiv(inner, BLOCK), n_channels, outer).
    pid0 = tl.program_id(0)
    c = tl.program_id(1)
    o = tl.program_id(2)

    idx = pid0 * BLOCK + tl.arange(0, BLOCK)
    valid = idx < inner
    off = (o * n_channels + c) * inner + idx

    x = tl.load(input_ptr + off, mask=valid, other=0.0)
    scale = tl.load(scale_ptr + c)
    zero_point = tl.load(zero_point_ptr + c)

    x_fp32 = x.to(tl.float32)
    scale_fp32 = scale.to(tl.float32)
    zero_point_fp32 = zero_point.to(tl.float32)

    nudge = _round_half_to_even(x_fp32 / scale_fp32) + zero_point_fp32
    cm_f = _mask_f32(nudge, quant_min, quant_max)
    nudge = tl.minimum(tl.maximum(nudge, quant_min), quant_max)
    output = (nudge - zero_point_fp32) * scale_fp32

    tl.store(output_ptr + off, output, mask=valid)
    tl.store(cachemask_fp32_ptr + off, cm_f, mask=valid)


@triton.jit
def _fake_quantize_per_channel_affine_cachemask_cols_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    cachemask_fp32_ptr,
    n_channels,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
):
    # Grid: (cdiv(n_channels, BLOCK), outer).  Used when axis is the last
    # dimension (inner == 1).
    pid0 = tl.program_id(0)
    o = tl.program_id(1)

    idx = pid0 * BLOCK + tl.arange(0, BLOCK)
    valid = idx < n_channels
    off = o * n_channels + idx

    x = tl.load(input_ptr + off, mask=valid, other=0.0)
    scale = tl.load(scale_ptr + idx, mask=valid, other=1.0)
    zero_point = tl.load(zero_point_ptr + idx, mask=valid, other=0)

    x_fp32 = x.to(tl.float32)
    scale_fp32 = scale.to(tl.float32)
    zero_point_fp32 = zero_point.to(tl.float32)

    nudge = _round_half_to_even(x_fp32 / scale_fp32) + zero_point_fp32
    cm_f = _mask_f32(nudge, quant_min, quant_max)
    nudge = tl.minimum(tl.maximum(nudge, quant_min), quant_max)
    output = (nudge - zero_point_fp32) * scale_fp32

    tl.store(output_ptr + off, output, mask=valid)
    tl.store(cachemask_fp32_ptr + off, cm_f, mask=valid)


@triton.jit
def _fake_quantize_per_channel_affine_cachemask_rows_i1_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    cachemask_ptr,
    inner,
    n_channels,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
):
    # Single-launch path for tiny inputs (n <= _TINY_N): store the i1 mask
    # directly (no extra _copy_from launch).
    pid0 = tl.program_id(0)
    c = tl.program_id(1)
    o = tl.program_id(2)

    idx = pid0 * BLOCK + tl.arange(0, BLOCK)
    valid = idx < inner
    off = (o * n_channels + c) * inner + idx

    x = tl.load(input_ptr + off, mask=valid, other=0.0)
    scale = tl.load(scale_ptr + c)
    zero_point = tl.load(zero_point_ptr + c)

    x_fp32 = x.to(tl.float32)
    scale_fp32 = scale.to(tl.float32)
    zero_point_fp32 = zero_point.to(tl.float32)

    nudge = _round_half_to_even(x_fp32 / scale_fp32) + zero_point_fp32
    cachemask = (nudge >= quant_min) & (nudge <= quant_max)
    nudge = tl.minimum(tl.maximum(nudge, quant_min), quant_max)
    output = (nudge - zero_point_fp32) * scale_fp32

    tl.store(output_ptr + off, output, mask=valid)
    tl.store(cachemask_ptr + off, cachemask, mask=valid)


@triton.jit
def _fake_quantize_per_channel_affine_cachemask_cols_i1_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    cachemask_ptr,
    n_channels,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
):
    # Single-launch path for tiny inputs (n <= _TINY_N, axis == last dim).
    pid0 = tl.program_id(0)
    o = tl.program_id(1)

    idx = pid0 * BLOCK + tl.arange(0, BLOCK)
    valid = idx < n_channels
    off = o * n_channels + idx

    x = tl.load(input_ptr + off, mask=valid, other=0.0)
    scale = tl.load(scale_ptr + idx, mask=valid, other=1.0)
    zero_point = tl.load(zero_point_ptr + idx, mask=valid, other=0)

    x_fp32 = x.to(tl.float32)
    scale_fp32 = scale.to(tl.float32)
    zero_point_fp32 = zero_point.to(tl.float32)

    nudge = _round_half_to_even(x_fp32 / scale_fp32) + zero_point_fp32
    cachemask = (nudge >= quant_min) & (nudge <= quant_max)
    nudge = tl.minimum(tl.maximum(nudge, quant_min), quant_max)
    output = (nudge - zero_point_fp32) * scale_fp32

    tl.store(output_ptr + off, output, mask=valid)
    tl.store(cachemask_ptr + off, cachemask, mask=valid)


def _launch_rows(
    input,
    scale,
    zero_point,
    output,
    cachemask,
    inner,
    n_channels,
    outer,
    quant_min,
    quant_max,
    mask_fp32=None,
):
    # BLOCK grows with inner (power of two, capped at 8192): a hard-coded
    # 1024-lane block keeps every program a toy row; 8192/8w is the measured
    # sweet spot for large inner sizes.
    block = 1024
    while block * 2 <= inner and block < 8192:
        block *= 2
    num_warps = 8 if block >= 2048 else 4
    grid = (triton.cdiv(inner, block), n_channels, outer)
    with torch_device_fn.device(input.device):
        if mask_fp32 is None:
            _fake_quantize_per_channel_affine_cachemask_rows_i1_kernel[grid](
                input,
                scale,
                zero_point,
                output,
                cachemask,
                inner,
                n_channels,
                quant_min,
                quant_max,
                BLOCK=block,
                num_warps=num_warps,
            )
        else:
            _fake_quantize_per_channel_affine_cachemask_rows_kernel[grid](
                input,
                scale,
                zero_point,
                output,
                mask_fp32,
                inner,
                n_channels,
                quant_min,
                quant_max,
                BLOCK=block,
                num_warps=num_warps,
            )


def _launch_cols(
    input,
    scale,
    zero_point,
    output,
    cachemask,
    n_channels,
    outer,
    quant_min,
    quant_max,
    mask_fp32=None,
):
    block = 1024
    while block * 2 <= n_channels and block < 8192:
        block *= 2
    num_warps = 8 if block >= 2048 else 4
    grid = (triton.cdiv(n_channels, block), outer)
    with torch_device_fn.device(input.device):
        if mask_fp32 is None:
            _fake_quantize_per_channel_affine_cachemask_cols_i1_kernel[grid](
                input,
                scale,
                zero_point,
                output,
                cachemask,
                n_channels,
                quant_min,
                quant_max,
                BLOCK=block,
                num_warps=num_warps,
            )
        else:
            _fake_quantize_per_channel_affine_cachemask_cols_kernel[grid](
                input,
                scale,
                zero_point,
                output,
                mask_fp32,
                n_channels,
                quant_min,
                quant_max,
                BLOCK=block,
                num_warps=num_warps,
            )


def _fake_quantize_per_channel_affine_cachemask_impl(
    input,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    output=None,
    cachemask=None,
):
    orig_dtype = input.dtype
    input = input.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    # fp16 in-kernel load + cast is miscompiled (KERNEL_OPT_EXPERIENCE case 11);
    # pre-convert to fp32 only for large n_channels (small n_channels are
    # launch-bound, where the conversion overhead outweighs the benefit).
    n_channels = input.shape[axis]
    compute_in_fp32 = (input.dtype == torch.float16) and (n_channels >= 64)
    if compute_in_fp32:
        input = input.to(torch.float32)

    output_preallocated = output
    if output is None or compute_in_fp32:
        output = torch.empty_like(input)
    if cachemask is None:
        cachemask = torch.empty_like(input, dtype=torch.bool)

    n_elements = input.numel()
    if n_elements == 0:
        if compute_in_fp32:
            if output_preallocated is None:
                output = output.to(orig_dtype)
            else:
                output_preallocated.copy_(output)
                output = output_preallocated
        return output, cachemask

    inner = 1
    for size in input.shape[axis + 1 :]:
        inner *= size

    with torch_device_fn.device(input.device):
        if inner > 1:
            outer = n_elements // (n_channels * inner)
            if n_elements <= _TINY_N:
                _launch_rows(
                    input,
                    scale,
                    zero_point,
                    output,
                    cachemask,
                    inner,
                    n_channels,
                    outer,
                    quant_min,
                    quant_max,
                )
            else:
                mask_fp32 = torch.empty(
                    input.shape, dtype=torch.float32, device=input.device
                )
                _launch_rows(
                    input,
                    scale,
                    zero_point,
                    output,
                    cachemask,
                    inner,
                    n_channels,
                    outer,
                    quant_min,
                    quant_max,
                    mask_fp32=mask_fp32,
                )
                # Vendor DMA converter fp32 -> bool; exact for {0.0, 1.0}
                # masks (verified: same as reference for 0/1 inputs).
                torch.ops.aten._copy_from(mask_fp32, cachemask, False)
        else:
            outer = n_elements // n_channels
            if n_elements <= _TINY_N:
                _launch_cols(
                    input,
                    scale,
                    zero_point,
                    output,
                    cachemask,
                    n_channels,
                    outer,
                    quant_min,
                    quant_max,
                )
            else:
                mask_fp32 = torch.empty(
                    input.shape, dtype=torch.float32, device=input.device
                )
                _launch_cols(
                    input,
                    scale,
                    zero_point,
                    output,
                    cachemask,
                    n_channels,
                    outer,
                    quant_min,
                    quant_max,
                    mask_fp32=mask_fp32,
                )
                torch.ops.aten._copy_from(mask_fp32, cachemask, False)
    if compute_in_fp32:
        if output_preallocated is None:
            output = output.to(orig_dtype)
        else:
            output_preallocated.copy_(output)
            output = output_preallocated
    return output, cachemask


def fake_quantize_per_channel_affine_cachemask(
    input, scale, zero_point, axis, quant_min, quant_max
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input, scale, zero_point, axis, quant_min, quant_max
    )


# --- Out variant (integrated passed22 candidate) -----------------------------
# Semantics reproduced from the ATen reference (verified on this target):
#   q   = nearbyint_(half-to-even)(x / scale) + zero_point   # fp32 opmath
#   m   = (q >= quant_min) & (q <= quant_max)                # inclusive
#   out = (clamp(q, quant_min, quant_max) - zero_point) * scale
# The quotient must match the ATen CPU reference: measured tie cases show it
# uses a correctly rounded reciprocal multiply ``x * (1 / scale)`` (the master
# CPU reference and the old kunlunxin path agree on all 48 diagnostic cases),
# so the kernel uses the same form. Ties round half-to-even (no native rint on
# this backend) via an explicit floor/fraction RNE that also defeats FMA
# contraction of the product into the rounding add.
# Both destinations (out0, out1) are written in place and returned.

_OUT_NDIM_MAX = 8


@triton.jit
def _out_round_half_to_even(x):
    """round-to-nearest, ties-to-even (matches std::nearbyint in fp32).

    The fp32 magic-constant add is deliberately not used: on this backend the
    compiler contracts ``(x * inv + magic)`` into an FMA, which skips the
    intermediate rounding of the product and changes half-way results (measured
    one-step divergence from the ATen reference). The explicit floor/fraction
    form forces the product to be materialised and is exact for all magnitudes.
    """
    fl = tl.floor(x)
    frac = x - fl
    is_even = (tl.floor(fl * 0.5) * 2.0) == fl
    return tl.where((frac > 0.5) | ((frac == 0.5) & (is_even == 0)), fl + 1.0, fl)


@triton.jit
def _fq_cachemask_kernel(
    x_ptr,
    s_ptr,
    z_ptr,
    o_ptr,
    m_ptr,
    n_elements,
    qmin,
    qmax,
    INNER: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    FULL: tl.constexpr,
    MB: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if INNER == 1:
        ch = off % C
    else:
        ch = (off // INNER) % C

    if FULL:
        # n_elements is an exact multiple of BLOCK: no lane is ever out of
        # bounds, so neither the loads nor the stores need predication.
        s = tl.load(s_ptr + ch).to(tl.float32)
        z = tl.load(z_ptr + ch).to(tl.float32)
        x = tl.load(x_ptr + off).to(tl.float32)
        inv = tl.div_rn(1.0, s)
        q = _out_round_half_to_even(x * inv) + z
        qc = tl.minimum(tl.maximum(q, qmin), qmax)
        # in-range == clamp did not change the value (identical to the inclusive
        # qmin/qmax test, and False for NaN), so reuse the clamp result.
        if MB:
            tl.store(m_ptr + off, q == qc)
        else:
            tl.store(m_ptr + off, (q == qc).to(tl.uint8))
        tl.store(o_ptr + off, ((qc - z) * s).to(o_ptr.dtype.element_ty))
    else:
        ok = off < n_elements
        s = tl.load(s_ptr + ch, mask=ok, other=1.0).to(tl.float32)
        z = tl.load(z_ptr + ch, mask=ok, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=ok, other=0.0).to(tl.float32)
        inv = tl.div_rn(1.0, s)
        q = _out_round_half_to_even(x * inv) + z
        qc = tl.minimum(tl.maximum(q, qmin), qmax)
        if MB:
            tl.store(m_ptr + off, q == qc, mask=ok)
        else:
            tl.store(m_ptr + off, (q == qc).to(tl.uint8), mask=ok)
        tl.store(o_ptr + off, ((qc - z) * s).to(o_ptr.dtype.element_ty), mask=ok)


@triton.jit
def _fq_cachemask_strided_kernel(
    x_ptr,
    s_ptr,
    z_ptr,
    o_ptr,
    m_ptr,
    n_elements,
    inner,
    C,
    qmin,
    qmax,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    ms0,
    ms1,
    ms2,
    ms3,
    ms4,
    ms5,
    ms6,
    ms7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    ok = off < n_elements
    ch = (off // inner) % C
    s = tl.load(s_ptr + ch, mask=ok, other=1.0).to(tl.float32)
    z = tl.load(z_ptr + ch, mask=ok, other=0.0).to(tl.float32)
    inv = tl.div_rn(1.0, s)

    xo = tl.zeros([BLOCK], dtype=tl.int64)
    oo = tl.zeros([BLOCK], dtype=tl.int64)
    mo = tl.zeros([BLOCK], dtype=tl.int64)
    rem = off
    if NDIM > 7:
        i7 = rem % d7
        rem = rem // d7
        xo += i7 * xs7
        oo += i7 * os7
        mo += i7 * ms7
    if NDIM > 6:
        i6 = rem % d6
        rem = rem // d6
        xo += i6 * xs6
        oo += i6 * os6
        mo += i6 * ms6
    if NDIM > 5:
        i5 = rem % d5
        rem = rem // d5
        xo += i5 * xs5
        oo += i5 * os5
        mo += i5 * ms5
    if NDIM > 4:
        i4 = rem % d4
        rem = rem // d4
        xo += i4 * xs4
        oo += i4 * os4
        mo += i4 * ms4
    if NDIM > 3:
        i3 = rem % d3
        rem = rem // d3
        xo += i3 * xs3
        oo += i3 * os3
        mo += i3 * ms3
    if NDIM > 2:
        i2 = rem % d2
        rem = rem // d2
        xo += i2 * xs2
        oo += i2 * os2
        mo += i2 * ms2
    if NDIM > 1:
        i1 = rem % d1
        rem = rem // d1
        xo += i1 * xs1
        oo += i1 * os1
        mo += i1 * ms1
    i0 = rem % d0
    xo += i0 * xs0
    oo += i0 * os0
    mo += i0 * ms0

    x = tl.load(x_ptr + xo, mask=ok, other=0.0).to(tl.float32)
    q = _out_round_half_to_even(x * inv) + z
    qc = tl.minimum(tl.maximum(q, qmin), qmax)
    tl.store(m_ptr + mo, (q == qc).to(tl.uint8), mask=ok)
    tl.store(o_ptr + oo, ((qc - z) * s).to(o_ptr.dtype.element_ty), mask=ok)


def _out_run(
    input,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    *,
    out0,
    out1,
):
    n_elements = input.numel()
    if n_elements == 0:
        return out0, out1

    nd = input.dim()
    a = axis + nd if axis < 0 else axis
    shape = input.shape
    C = shape[a]
    inner = 1
    for d in shape[a + 1 :]:
        inner *= d

    if out1.dtype == torch.bool:
        mask_bool = True
        mask_arg = out1
    else:
        mask_bool = False
        mask_arg = out1.view(torch.uint8)

    qmin = float(quant_min)
    qmax = float(quant_max)

    if input.is_contiguous() and out0.is_contiguous() and mask_arg.is_contiguous():
        # Cost on this target tracks CTAs and tile bytes: the measured optimum
        # elements-per-CTA grows with the problem size (98304 -> 4096 elements,
        # >=524288 -> 8192), while launch-bound tiny inputs keep the smallest
        # tile.
        if n_elements <= 8192:
            BLOCK = 1024
            nw = 4
        elif n_elements <= 131072:
            BLOCK = 4096
            nw = 4
        else:
            BLOCK = 8192
            nw = 4
        with torch_device_fn.device(input.device):
            _fq_cachemask_kernel[(triton.cdiv(n_elements, BLOCK),)](
                input,
                scale,
                zero_point,
                out0,
                mask_arg,
                n_elements,
                qmin,
                qmax,
                INNER=inner,
                C=C,
                BLOCK=BLOCK,
                FULL=(n_elements % BLOCK == 0),
                MB=mask_bool,
                num_warps=nw,
            )
        return out0, out1

    # general strided fallback (no layout conversion on the data path)
    def _pad(seq):
        seq = list(seq) + [1] * (_OUT_NDIM_MAX - len(seq))
        return seq

    ds = _pad(shape)
    xs = _pad(input.stride())
    os_ = _pad([s // out0.element_size() for s in out0.stride()])
    mask_view = out1.view(torch.uint8) if mask_bool else out1
    ms = _pad(mask_view.stride())
    BLOCK = 1024
    with torch_device_fn.device(input.device):
        _fq_cachemask_strided_kernel[(triton.cdiv(n_elements, BLOCK),)](
            input,
            scale,
            zero_point,
            out0,
            mask_view,
            n_elements,
            inner,
            C,
            qmin,
            qmax,
            *ds,
            *xs,
            *os_,
            *ms,
            NDIM=nd,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out0, out1


def fake_quantize_per_channel_affine_cachemask_out(
    input, scale, zero_point, axis, quant_min, quant_max, *, out0, out1
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_OUT")
    return _out_run(
        input, scale, zero_point, axis, quant_min, quant_max, out0=out0, out1=out1
    )
