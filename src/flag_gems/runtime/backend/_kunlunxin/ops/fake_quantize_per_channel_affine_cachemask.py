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

Design notes (vs. the generic 1D implementation, ~0.8 ms/kernel + ~236 ms/call
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
    input = input.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    if output is None:
        output = torch.empty_like(input)
    if cachemask is None:
        cachemask = torch.empty_like(input, dtype=torch.bool)

    n_elements = input.numel()
    if n_elements == 0:
        return output, cachemask

    n_channels = input.shape[axis]
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
                    n_elements, dtype=torch.float32, device=input.device
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
                    n_elements, dtype=torch.float32, device=input.device
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
    return output, cachemask


def fake_quantize_per_channel_affine_cachemask(
    input, scale, zero_point, axis, quant_min, quant_max
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input, scale, zero_point, axis, quant_min, quant_max
    )


def fake_quantize_per_channel_affine_cachemask_out(
    input, scale, zero_point, axis, quant_min, quant_max, *, out0, out1
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_CHANNEL_AFFINE_CACHEMASK_OUT")
    return _fake_quantize_per_channel_affine_cachemask_impl(
        input,
        scale,
        zero_point,
        axis,
        quant_min,
        quant_max,
        output=out0,
        cachemask=out1,
    )
