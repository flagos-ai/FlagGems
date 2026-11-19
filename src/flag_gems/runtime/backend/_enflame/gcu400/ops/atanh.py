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

"""GCU400-specific implementation of atanh / atanh_: 1D flat Triton kernel.

Input and output may share one memory block (in-place) or use a separate output (out-of-place);
large-shape tiles are chosen by element bytes, and resident CTAs are reused with a grid-stride loop.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Resident CTA count matches GCU400 SIP topology; large-shape tiles are tiered by element bytes
NUM_CTAS = 48
BLOCK_LARGE_16BIT = 8192
BLOCK_LARGE_32BIT = 8192
NUM_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def atanh_kernel(
    x_ptr,
    out_ptr,
    num_elements,
    BLOCK: tl.constexpr,
    FORMULA_MODE: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    MULTIPLE_OF_HINT: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = (num_elements + BLOCK - 1) // BLOCK
    for tile_index in tl.range(
        program_id, num_tiles, num_programs, num_stages=LOOP_STAGES
    ):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        if MULTIPLE_OF_HINT == 1:
            offsets = tl.multiple_of(offsets, BLOCK)
        x = tl.load(x_ptr + offsets, mask=offsets < num_elements, other=0.0)
        if COMPUTE_MODE == 0:
            value = x.to(tl.float32)
        else:
            value = x
        if FORMULA_MODE == 0:
            # Original definition: atanh(x) = 0.5 * log((1+x)/(1-x))
            result = 0.5 * tl.log((1.0 + value) / (1.0 - value))
        elif FORMULA_MODE == 1:
            # Subtract two logs: near +/-1 this avoids error amplification from rounding the division quotient
            result = 0.5 * (tl.log(1.0 + value) - tl.log(1.0 - value))
        elif FORMULA_MODE == 2:
            # log2 equivalent rewrite, log(x) = log2(x) * ln(2)
            result = 0.34657359027997264 * tl.log2((1.0 + value) / (1.0 - value))
        else:
            # Single division: atanh(x) = 0.5 * log(1 + 2x/(1-x))
            result = 0.5 * tl.log(1.0 + 2.0 * value / (1.0 - value))
        tl.store(out_ptr + offsets, result, mask=offsets < num_elements)


def _choose_block(num_elements, is_fp32):
    """Small shapes use 1024, medium shapes the nearest power of two, large shapes are tiered by dtype."""
    if num_elements <= 1024:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    return BLOCK_LARGE_32BIT if is_fp32 else BLOCK_LARGE_16BIT


def _launch(inp, out):
    num_elements = inp.numel()
    if num_elements == 0:
        return out
    is_fp32 = inp.dtype == torch.float32
    block_size = _choose_block(num_elements, is_fp32)
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    with torch_device_fn.device(inp.device):
        atanh_kernel[(grid_size,)](
            inp,
            out,
            num_elements,
            BLOCK=block_size,
            FORMULA_MODE=0,
            COMPUTE_MODE=0,
            MULTIPLE_OF_HINT=0,
            LOOP_STAGES=1,
            num_warps=NUM_WARPS,
        )
    return out


def _atanh(A):
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


def atanh(A):
    logger.debug("GEMS_ENFLAME ATANH")
    return _atanh(A)


def atanh_(A):
    logger.debug("GEMS_ENFLAME ATANH_")
    if not A.is_contiguous():
        A.copy_(_atanh(A))
        return A
    return _launch(A, A)
