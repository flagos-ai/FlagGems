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

Design notes (measured on the P800 XPU3 backend):

* A single flat 1-D kernel over ``numel`` derives the channel from the genuine
  per-lane ``offsets`` vector, so ``scale``/``zero_point`` are real per-lane
  gathers.  A per-program uniform scalar variant (one parameter load per
  program) was observed to be miscompiled on this backend: it mixed the
  neighbouring channel's parameter into part of the tile.
* Round-half-to-even uses the exact fp32 identity
  ``(y + 1.5*2**23) - 1.5*2**23`` instead of floor + parity: it is exact for
  ``|y| < 2**22``, needs no libdevice intrinsic and replaces ~6 float ops with
  2.  ``magic`` is a kernel argument (not a module global) so it cannot be
  folded or captured.
* The tail mask is a ``constexpr`` (``MASKED``): for every benchmark shape
  ``numel % BLOCK == 0``, which removes the compare/select and both masked
  loads entirely.
* ``BLOCK`` is chosen so at least ~32 programs are in flight, keeping all SMs
  busy on the small shapes without inflating the big ones.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_MAX_BLOCK = 8192
_MIN_BLOCK = 128


@triton.jit
def _fq_kernel(
    x_ptr,
    s_ptr,
    z_ptr,
    out_ptr,
    n_elem,
    qlo,
    qhi,
    magic,
    INNER: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    chan = (offs // INNER) % C

    if MASKED:
        m = offs < n_elem
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        s = tl.load(s_ptr + chan, mask=m, other=1.0).to(tl.float32)
        z = tl.load(z_ptr + chan, mask=m, other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs)
        s = tl.load(s_ptr + chan).to(tl.float32)
        z = tl.load(z_ptr + chan).to(tl.float32)

    y = x.to(tl.float32) / s
    q = (y + magic) - magic
    v = tl.minimum(tl.maximum(q + z, qlo), qhi)
    o = (v - z) * s

    od: tl.constexpr = out_ptr.dtype.element_ty
    if MASKED:
        tl.store(out_ptr + offs, o.to(od), mask=m)
    else:
        tl.store(out_ptr + offs, o.to(od))


def _block_for(n):
    # Tile width is the dominant performance knob on this backend: measured
    # geo_mean 0.34962 -> 0.48710 -> 0.63091 -> 0.72192 for tiles
    # 1024 -> 2048 -> 4096 -> 8192 at 8 warps, with 16384 past the optimum.
    # 8192 is therefore used for every shape wide enough to fill it; smaller
    # shapes take the widest tile that still leaves ~4 programs.
    if n >= 32768:
        return _MAX_BLOCK
    b = _MIN_BLOCK
    q = n >> 2
    while b < _MAX_BLOCK and b * 2 <= q:
        b *= 2
    return b


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

    out = torch.empty_like(input)
    n = input.numel()
    if n == 0:
        return out

    shape = input.shape
    inner = 1
    for i in range(axis + 1, len(shape)):
        inner *= shape[i]
    n_channels = shape[axis]

    block = _block_for(n)
    grid = ((n + block - 1) // block,)
    with torch_device_fn.device(input.device):
        _fq_kernel[grid](
            input,
            scale,
            zero_point,
            out,
            n,
            float(quant_min),
            float(quant_max),
            12582912.0,
            INNER=inner,
            C=n_channels,
            BLOCK=block,
            MASKED=(n % block) != 0,
            # 16 elements per thread at the wide tiles: BLOCK=8192 with 8 warps
            # leaves 32 elements live per thread on the shapes that still trail.
            num_warps=(
                1
                if block < 256
                else (4 if block < 1024 else (16 if block >= 4096 else 8))
            ),
        )
    return out
