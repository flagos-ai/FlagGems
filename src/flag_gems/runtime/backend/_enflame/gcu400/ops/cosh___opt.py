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

"""GCU400-specific Triton implementation of cosh_: in-place elementwise hyperbolic cosine.

A 1D flat kernel with a fixed number of CTAs and a grid-stride loop replaces the common pointwise_dynamic
codegen's Python-side dispatch; input and output share one memory block, with one read and one write each.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cosh import cosh_func as _common_cosh_func
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Resident CTA count aligned with GCU400 SIP topology, reused by an in-kernel stride loop
NUM_CTAS = 48
# Tile element count tiered by element bytes for large shapes
BLOCK_LARGE_16BIT = 8192
BLOCK_LARGE_32BIT = 32768
NUM_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _cosh_inplace_kernel(x_ptr, num_elements, BLOCK: tl.constexpr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(num_elements, BLOCK)
    for tile_index in tl.range(program_id, num_tiles, num_programs):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_elements
        value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # cosh(x) = (exp(x) + exp(-x)) / 2; fp32 intermediates preserve accuracy for low-precision inputs
        result = 0.5 * (tl.exp(value) + tl.exp(-value))
        tl.store(x_ptr + offsets, result, mask=mask)


def _select_block_size(num_elements, is_fp32):
    """Small shapes use 1024, medium shapes the nearest power of two, large shapes are tiered by dtype."""
    if num_elements <= 1024:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    return BLOCK_LARGE_32BIT if is_fp32 else BLOCK_LARGE_16BIT


def cosh_(A):
    logger.debug("GEMS_ENFLAME COSH_")
    num_elements = A.numel()
    if num_elements == 0:
        return A
    # Non-float / non-contiguous inputs keep the common codegen path, preserving promotion and stride semantics
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if A.dtype not in supported_dtypes or not A.is_contiguous():
        return _common_cosh_func(A, out0=A)
    is_fp32 = A.dtype == torch.float32
    block_size = _select_block_size(num_elements, is_fp32)
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    with torch_device_fn.device(A.device):
        _cosh_inplace_kernel[(grid_size,)](
            A,
            num_elements,
            BLOCK=block_size,
            num_warps=NUM_WARPS,
        )
    return A


__all__ = ["cosh_"]
