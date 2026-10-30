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
"""fake_quantize_per_tensor_affine for Kunlunxin.

`tl_extra_shim.nearbyint` is evaluated one lane at a time on this backend.
Measured at n=1048576 (harness/scratch_fq/probe_fq1..3.py), same traffic, best
tile and warp count per row:

| form                                   | fp16    | fp32    |
|---|---|---|
| `store(load(x))`                       |  19.1us |  18.2us |
| `(x * inv_scale) * scale`              |  18.1us |  18.3us |
| whole chain with `nearbyint`           | 318.1us | 305.5us |
| whole chain with the magic add         |  18.4us |  18.3us |
| torch                                  |  13.7us |  15.6us |

So the rounding alone costs ~100us over the traffic (0.095 ns/elem, a per-lane
cost) and poisons everything downstream of it: the remaining adds, clamps and
multiplies push it to 318us, 17x the copy floor, while the same chain with a
vector round is *at* the copy floor and the arithmetic is free.

The round used here is the classic magic add: for `0 <= v < 2^22`,
`(v + 12582912.0) - 12582912.0` is exactly `nearbyint(v)`, because 12582912.0 is
1.5 * 2^23, `v + M` lands in the binade where the spacing is 1.0, and IEEE
round-to-nearest-even on that single add *is* ties-to-even.  The sign is handled
by signing M; the magnitude limit is handled by clamping `v` into
`[quant_min - zero_point - 1, quant_max - zero_point + 1]` *before* rounding.

That pre-clamp is exact rather than an approximation.  Rounding is monotone, so
any `v` above the window rounds to at least `quant_max - zero_point + 1`, which
saturates to `quant_max` after `+ zero_point` and the final clamp -- the same
value the clamped `v` produces; likewise below.  NaN propagates through
`tl.minimum`/`tl.maximum` on this backend, so NaN inputs are unaffected
(verified against the generic kernel).  It is only valid while the window fits
in +-2^22, which is where the magic add is exact, so wider quant ranges keep the
`nearbyint` kernel.

One pre-existing difference is preserved, not introduced: for `+inf` torch
returns `quant_min` (it casts to an integer and inherits the UB) while every
float-domain formulation -- including the generic `nearbyint` kernel this
replaces -- returns `quant_max`.
"""

import logging
import math
import struct

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

# `(v + M) - M` is exact ties-to-even only while `v + M` stays in the binade of
# M = 1.5 * 2^23, i.e. for |v| < 2^22.
_ROUND_SAFE = float(1 << 22)
_MAGIC = 12582912.0


@triton.jit(
    do_not_specialize=[
        "n_elements",
        "scale",
        "inv_scale",
        "zero_point",
        "quant_min",
        "quant_max",
        "lo",
        "hi",
    ]
)
def fake_quantize_per_tensor_affine_round_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale,
    inv_scale,
    zero_point,
    quant_min,
    quant_max,
    lo,
    hi,
    EVEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN:
        value = tl.load(input_ptr + offsets).to(tl.float32)
    else:
        value = tl.load(
            input_ptr + offsets, mask=offsets < n_elements, other=0.0
        ).to(tl.float32)
    value = value * inv_scale.to(tl.float32)
    # Pre-clamp so the magic add below stays in its exact range.
    value = tl.minimum(tl.maximum(value, lo.to(tl.float32)), hi.to(tl.float32))
    magic = tl.where(value < 0.0, -12582912.0, 12582912.0)
    quantized = ((value + magic) - magic) + zero_point
    quantized = tl.minimum(tl.maximum(quantized, quant_min), quant_max)
    output = ((quantized - zero_point) * scale.to(tl.float32)).to(
        output_ptr.dtype.element_ty
    )
    if EVEN:
        tl.store(output_ptr + offsets, output)
    else:
        tl.store(output_ptr + offsets, output, mask=offsets < n_elements)


@triton.jit(
    do_not_specialize=[
        "n_elements",
        "scale",
        "inv_scale",
        "zero_point",
        "quant_min",
        "quant_max",
    ]
)
def fake_quantize_per_tensor_affine_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale,
    inv_scale,
    zero_point,
    quant_min,
    quant_max,
    BLOCK_SIZE: tl.constexpr,
):
    """Quant range wider than +-2^22: fall back to the per-lane round."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    quantized = (
        tl_extra_shim.nearbyint(value * inv_scale.to(tl.float32)) + zero_point
    )
    quantized = tl.minimum(tl.maximum(quantized, quant_min), quant_max)
    output = (quantized - zero_point) * scale.to(tl.float32)
    tl.store(output_ptr + offsets, output, mask=mask)


# These shapes are launch-bound, not bandwidth-bound: the whole benchmark matrix
# tops out at 1048576 elements, where the traffic is ~2us and the measured time
# is ~18us.  Aim the tile at a grid of ~12 (= P800 cluster_num) so one wave
# covers the machine, with the tile bounded so tiny inputs do not pay for a
# 32768-lane program and huge ones do not serialize on a single CTA.
def _pick_block(n_elements):
    if n_elements <= 1024:
        return max(16, triton.next_power_of_2(n_elements))
    block = triton.next_power_of_2(triton.cdiv(n_elements, 12))
    return min(32768, max(1024, block))


_LAUNCH = dict(
    num_warps=8,
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseVectorization=False,
    isCloseMemoryAsync=False,
)


def fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max):
    logger.debug("GEMS_KUNLUNXIN FAKE_QUANTIZE_PER_TENSOR_AFFINE")

    input = input.contiguous()
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    if isinstance(zero_point, torch.Tensor):
        zero_point = zero_point.item()
    if quant_min > quant_max:
        raise RuntimeError("quant_min must be less than or equal to quant_max")
    if zero_point < quant_min or zero_point > quant_max:
        raise RuntimeError("zero_point must be between quant_min and quant_max")

    output = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return output

    scale = struct.unpack("f", struct.pack("f", float(scale)))[0]
    if scale == 0.0:
        inv_scale = math.copysign(math.inf, scale)
    else:
        inv_scale = struct.unpack("f", struct.pack("f", 1.0 / scale))[0]

    zero_point = int(zero_point)
    quant_min = int(quant_min)
    quant_max = int(quant_max)
    lo = float(quant_min - zero_point - 1)
    hi = float(quant_max - zero_point + 1)
    block_size = _pick_block(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(input.device):
        if -_ROUND_SAFE <= lo and hi <= _ROUND_SAFE:
            fake_quantize_per_tensor_affine_round_kernel[grid](
                input,
                output,
                n_elements,
                scale,
                inv_scale,
                zero_point,
                quant_min,
                quant_max,
                lo,
                hi,
                EVEN=n_elements % block_size == 0,
                BLOCK_SIZE=block_size,
                **_LAUNCH,
            )
        else:
            fake_quantize_per_tensor_affine_kernel[grid](
                input,
                output,
                n_elements,
                scale,
                inv_scale,
                zero_point,
                quant_min,
                quant_max,
                BLOCK_SIZE=block_size,
                **_LAUNCH,
            )
    return output
