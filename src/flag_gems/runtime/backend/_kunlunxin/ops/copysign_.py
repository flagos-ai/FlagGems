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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_INT_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


@libentry()
@triton.jit(do_not_specialize=["num_tasks"])
def _copysign_inplace_kernel(
    A, B, num_tasks, TILE: tl.constexpr, TILES_PER_CTA: tl.constexpr, ONE_TILE: tl.constexpr
):
    # Operates on integer views of the operands (see wrapper): pure sign-bit
    # manipulation, out = (|a| bits) | (sign bit of b). The generic pointwise
    # path promotes bf16/f16 inputs to fp32 and crashes in
    # TritonXPUDtypeConvert ("cannot bitcast size 32 to 16"); integer views
    # keep every access same-width and also avoid the fp abs/where cost.
    ity = A.type.element_ty
    num_bits: tl.constexpr = ity.primitive_bitwidth
    # signed-safe constants: the sign-bit-only value is -(1<<(w-1));
    # clear_mask = all bits except the sign bit.
    sign_mask: tl.constexpr = -(1 << (num_bits - 1))
    clear_mask: tl.constexpr = (1 << (num_bits - 1)) - 1

    pid = tl.program_id(0)
    if ONE_TILE:
        tid = pid * TILE + tl.arange(0, TILE)
        mask = tid < num_tasks
        a_bits = tl.load(A + tid, mask=mask)
        b_bits = tl.load(B + tid, mask=mask)
        out_bits = (a_bits & clear_mask) | (b_bits & sign_mask)
        tl.store(A + tid, out_bits, mask=mask)
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, TILES_PER_CTA):
            tile_id = pid + j * num_ctas
            tid = tile_id * TILE + tl.arange(0, TILE)
            mask = tid < num_tasks
            a_bits = tl.load(A + tid, mask=mask)
            b_bits = tl.load(B + tid, mask=mask)
            out_bits = (a_bits & clear_mask) | (b_bits & sign_mask)
            tl.store(A + tid, out_bits, mask=mask)


def _copysign_run(input, other):
    num_tasks = input.numel()
    if num_tasks == 0:
        return input
    ity = _INT_VIEW[input.element_size()]
    a = input.view(ity)
    b = other.view(ity) if other.dtype == input.dtype else other.to(input.dtype).view(ity)
    num_ctas = 12
    num_tiles = num_ctas
    tile = triton.next_power_of_2(triton.cdiv(num_tasks, num_tiles))
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    _copysign_inplace_kernel[(num_ctas, 1, 1)](
        a,
        b,
        num_tasks,
        TILE=tile,
        TILES_PER_CTA=tiles_per_cta,
        ONE_TILE=tiles_per_cta == 1,
    )
    return input


def copysign_(input, other):
    """In-place copysign specialized for XPU: integer sign-bit kernel."""
    logger.debug("GEMS_KUNLUNXIN COPYSIGN_")
    if not input.is_contiguous() or not other.is_contiguous():
        input_c = input.contiguous()
        other_c = other.contiguous()
        _copysign_run(input_c, other_c)
        input.copy_(input_c)
        return input
    return _copysign_run(input, other)
