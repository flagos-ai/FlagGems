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

"""Optimized Kunlunxin (XPU) Triton kernel for fake_quantize_per_channel_affine.

Design notes (evidence in
harness/solution/fake_quantize_per_channel_affine/README.md):

* ``grid`` is a constant tuple (never a ``lambda``); on the xpubin backend a
  per-call ``lambda`` grid participates in the compile-cache key and forces a
  full recompile on every launch (~200-450 ms/call, size-independent).  This
  alone was the dominant cost of the pre-optimization baseline.
* The channel index is ``(off // channel_stride) % C``.  ``C`` is passed as a
  ``tl.constexpr`` and, when ``channel_stride`` is a power of two, it is
  passed as the constexpr shift ``LOG2_CS`` so the kernel computes
  ``(off >> LOG2_CS) % C`` (shifts/ands only).  The equivalent
  ``(off // CS) % C`` with a constexpr ``CS`` was observed **deterministically
  miscompiling** on this backend (maxdiff ~1e8-1e9 on (2,3,4) axis=1), so only
  the explicit-shift form is used.  When a BLOCK-wide gather would be uniform
  (channel slice >= 256 and aligned), the channel kernel (``c =
  program_id(0)``, one scalar load per program) is used instead: the
  uniform-address gather/sync-load path is ~10-100x slower.
* Round-half-to-even is done with the magic number
  ``(x + 1.5*2**23) - 1.5*2**23``: exact for ``|x| < 2**22``; for larger
  ``|x|`` the clamp to ``[quant_min, quant_max]`` saturates identically, so
  the final output matches the reference bitwise on its own half-to-even
  inputs.
* The load is mask-free (address clamped in-bounds for the tail block) while
  the store always keeps its tail mask (address clamped + value masked in the
  tail block): the verified pattern that avoids the KL3 299 poison-card on
  unmasked stores.  Removing the store mask would be faster but is
  intentionally not attempted (see README "已知限制").
* All quantization math is carried out in fp32; the output is stored in the
  input dtype.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Exact RNE (round-half-to-even) for |x| < 2**22 (fp32).  For |x| >= 2**22 the
# result differs from rne(x) by at most a few units, but then x/scale is far
# outside [quant_min, quant_max] so the final clamp saturates identically.
_RNE_MAGIC = tl.constexpr(12582912.0)  # 1.5 * 2**23


@triton.jit
def _fq_pca_flat_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    n_elem,
    channel_stride,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
    C: tl.constexpr,
    LOG2_CS: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        # n_elem % BLOCK == 0: every lane is in bounds and valid -> plain
        # unmasked load (block-DMA path); the store mask below is provably
        # all-true, so the stored block can never go out of bounds.
        x = tl.load(x_ptr + off).to(tl.float32)
    else:
        oc = tl.minimum(off, n_elem - 1)
        valid = off < n_elem
        x = tl.where(valid, tl.load(x_ptr + oc).to(tl.float32), 0.0)
    if LOG2_CS >= 0:
        c = (off >> LOG2_CS) % C
    else:
        c = (off // channel_stride) % C
    s = tl.load(scale_ptr + c).to(tl.float32)
    z = tl.load(zp_ptr + c).to(tl.float32)

    xq = (x / s + _RNE_MAGIC) - _RNE_MAGIC
    xq = xq + z
    xq = tl.minimum(tl.maximum(xq, quant_min), quant_max)
    out = (xq - z) * s

    if EVEN:
        tl.store(out_ptr + off, out, mask=off < n_elem)
    else:
        tl.store(
            out_ptr + tl.minimum(off, n_elem - 1),
            tl.where(off < n_elem, out, 0.0),
            mask=off < n_elem,
        )


@triton.jit
def _fq_pca_channel_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    n_elem,
    channel_stride,
    outer_stride,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Used when the channel slice is aligned: c = program_id(0) is the channel
    # (no per-element integer division, no per-lane gather).
    c = tl.program_id(0)
    o = tl.program_id(1)
    b = tl.program_id(2)
    j = tl.arange(0, BLOCK)
    so = b * BLOCK + j
    base = o * outer_stride + c * channel_stride
    if EVEN:
        x = tl.load(x_ptr + (base + so)).to(tl.float32)
    else:
        v = so < channel_stride
        x = tl.where(
            v,
            tl.load(x_ptr + (base + tl.minimum(so, channel_stride - 1))).to(tl.float32),
            0.0,
        )
    s = tl.load(scale_ptr + c).to(tl.float32)
    z = tl.load(zp_ptr + c).to(tl.float32)

    xq = (x / s + _RNE_MAGIC) - _RNE_MAGIC
    xq = xq + z
    xq = tl.minimum(tl.maximum(xq, quant_min), quant_max)
    out = (xq - z) * s

    if EVEN:
        tl.store(out_ptr + (base + so), out, mask=so < channel_stride)
    else:
        tl.store(
            out_ptr + (base + tl.minimum(so, channel_stride - 1)),
            tl.where(v, out, 0.0),
            mask=v,
        )


def _pick_block(n_elem):
    # Fixed, bounded tier table (no autotune).  Measured on P800 xpu3: small
    # shapes (<=16K elems) favour 512-2048, large shapes improve toward
    # 8192-16384 (one 16384-lane block covers a full 64KB cache line run).
    if n_elem <= 1024:
        return 512
    if n_elem <= 16384:
        return 2048
    if n_elem <= 131072:
        return 8192
    return 16384


def fake_quantize_per_channel_affine(
    input, scale, zero_point, axis, quant_min, quant_max
):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_CHANNEL_AFFINE")

    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    if axis < 0:
        axis += input.dim()

    input = input.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    n_elem = input.numel()
    if n_elem == 0:
        return torch.empty_like(input)

    shape = input.shape
    n_channels = shape[axis]

    channel_stride = 1
    for i in range(axis + 1, len(shape)):
        channel_stride *= shape[i]

    output = torch.empty_like(input)
    block = _pick_block(n_elem)
    # A BLOCK-wide gather whose per-lane index is uniform (channel slice >=
    # BLOCK) falls into the slow uniform/sync-load path on this backend; use
    # the channel kernel (c = program_id, one scalar load per program) there.
    ch_block = None
    if channel_stride >= 256 and channel_stride % 4096 == 0:
        ch_block = 16384 if channel_stride % 16384 == 0 else 4096
    elif channel_stride >= 256 and channel_stride % 1024 == 0:
        ch_block = 1024
    elif channel_stride % 256 == 0:
        ch_block = channel_stride  # 256 or 512
    if ch_block is not None:
        outer_stride = n_channels * channel_stride
        n_outer = n_elem // outer_stride
        grid = (n_channels, n_outer, channel_stride // ch_block)
        with torch_device_fn.device(input.device):
            _fq_pca_channel_kernel[grid](
                input,
                scale,
                zero_point,
                output,
                n_elem,
                channel_stride,
                outer_stride,
                quant_min,
                quant_max,
                BLOCK=ch_block,
                EVEN=True,
                num_warps=4,
            )
        return output

    is_pow2 = (channel_stride & (channel_stride - 1)) == 0
    log2_cs = channel_stride.bit_length() - 1 if is_pow2 else -1
    even = n_elem % block == 0
    grid = (triton.cdiv(n_elem, block),)
    with torch_device_fn.device(input.device):
        _fq_pca_flat_kernel[grid](
            input,
            scale,
            zero_point,
            output,
            n_elem,
            channel_stride,
            quant_min,
            quant_max,
            BLOCK=block,
            C=n_channels,
            LOG2_CS=log2_cs,
            EVEN=even,
            num_warps=4,
        )

    return output
