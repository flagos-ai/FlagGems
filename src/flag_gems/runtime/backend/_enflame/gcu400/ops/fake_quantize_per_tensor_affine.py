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

"""Per-tensor affine fake quantization (Triton).

Matches the ATen reference bit-exactly (the pytest asserts on exact equality):

    inv  = float32(1.0 / float32(scale))
    r    = nearbyint(float32(x * inv))            # round half to even
    q    = clamp(r + zero_point, quant_min, quant_max)
    out  = (q - zero_point) * float32(scale)

Equivalences used here (all exact in fp32 for the quant ranges used):

  * clamp(r + zp, qmin, qmax) - zp == clamp(r, qmin - zp, qmax - zp): the shift
    by an exact integer is monotone, so the two clamps select the same element.
    This removes one add and one subtract per element.
  * The magic-add rounding is evaluated on |p| and the sign restored.  On this
    backend the fused `(p + 2**23) - 2**23` elementwise chain is *not*
    round-half-to-even for negative p (it behaves like "round to a multiple of
    0.5", e.g. p=-22.398 -> -22.5), while the |p| form is exact.  Hence abs().

The whole measured workload is launch bound (empty kernel ~= 11.2us +
0.46us * grid.x on this vendor), so the tuning is entirely "finish the tensor
with as few programs as possible": one program covers the tensor up to 128K
elements, and num_warps is picked from the resulting grid.
"""

import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAGIC_C = tl.constexpr(8388608.0)  # 2**23
_BLOCK_MAX = 131072  # grid.x stays far below 65535 for any real tensor


@triton.jit
def fake_quantize_per_tensor_affine_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    inv,
    scale,
    lo,
    hi,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    p = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv

    # nearbyint(p), ties to even, on the magnitude: |p| + 2**23 lands in
    # [2**23, 2**24) where the ulp is one, so the hardware add rounds |p| to an
    # integer (round half to even).  |p| >= 2**23 is already integral and any
    # error there is absorbed by the clamp (quant ranges are far below 2**23).
    ap = tl.abs(p)
    r = (ap + _MAGIC_C) - _MAGIC_C
    r = tl.where(p < 0.0, -r, r)

    r = tl.minimum(tl.maximum(r, lo), hi)
    tl.store(out_ptr + offs, (r * scale).to(out_ptr.dtype.element_ty), mask=mask)


_pack = struct.pack
_unpack = struct.unpack


def _f32(v):
    return _unpack("f", _pack("f", v))[0]


def _num_warps(block, grid):
    if block <= 16384:
        return 2
    if block <= 32768:
        return 4
    if grid <= 2:
        return 8
    if grid <= 4:
        return 4
    return 2


def fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE")

    if type(scale) is not float:
        scale = scale.item() if isinstance(scale, torch.Tensor) else float(scale)
    if type(zero_point) is not int:
        zero_point = (
            zero_point.item()
            if isinstance(zero_point, torch.Tensor)
            else int(zero_point)
        )
    if type(quant_min) is not int:
        quant_min = (
            quant_min.item() if isinstance(quant_min, torch.Tensor) else int(quant_min)
        )
    if type(quant_max) is not int:
        quant_max = (
            quant_max.item() if isinstance(quant_max, torch.Tensor) else int(quant_max)
        )

    if quant_min > quant_max:
        raise RuntimeError("quant_min must be less than or equal to quant_max")
    if zero_point < quant_min or zero_point > quant_max:
        raise RuntimeError("zero_point must be between quant_min and quant_max")

    x = input if input.is_contiguous() else input.contiguous()

    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    s32 = _f32(scale)
    inv = _f32(1.0 / s32)

    block = 1 << max(10, (n - 1).bit_length())
    if block > _BLOCK_MAX:
        block = _BLOCK_MAX
    grid = (triton.cdiv(n, block),)

    fake_quantize_per_tensor_affine_kernel[grid](
        x,
        out,
        n,
        inv,
        s32,
        quant_min - zero_point + 0.0,
        quant_max - zero_point + 0.0,
        BLOCK=block,
        num_warps=_num_warps(block, grid[0]),
    )
    return out
