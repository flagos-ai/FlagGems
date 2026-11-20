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

"""GCU400-specific implementation of erf / erf_.

Non-in-place erf keeps the backend's pointwise_dynamic codegen semantics; in-place erf_ uses a fixed number of
CTAs + grid-stride 1D flat kernel, removing codegen's Python-side dispatch and index decomposition overhead.
Input and output share one memory block, with one read and one write per element.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Resident CTA count aligned with GCU400 SIP topology, reused by an in-kernel stride loop
NUM_CTAS = 24
# Tile element count tiered by element bytes for large shapes
BLOCK_LARGE_16BIT = 8192
BLOCK_LARGE_32BIT = 65536
NUM_WARPS = 4
MATH_MODE = 0

_extern_erf = tl_extra_shim.erf


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erf_func(x):
    return tl.math.erf(x.to(tl.float32))


def erf(x):
    logger.debug("GEMS_ENFLAME ERF")
    return erf_func(x)


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _erf_inplace_kernel(
    x_ptr, num_elements, BLOCK: tl.constexpr, MATH_MODE: tl.constexpr
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(num_elements, BLOCK)
    for tile_index in tl.range(program_id, num_tiles, num_programs):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_elements
        value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if MATH_MODE == 0:
            # Triton built-in erf, fully equivalent in semantics to the common codegen
            result = tl.math.erf(value)
        elif MATH_MODE == 1:
            # Device-side extern __ef_erff
            result = _extern_erf(value)
        elif MATH_MODE == 2:
            # Abramowitz-Stegun 7.1.26: one exp + 5 Horner multiply-adds, max error about 1.5e-7
            sign = tl.where(value >= 0.0, 1.0, -1.0)
            abs_value = tl.abs(value)
            reciprocal = 1.0 / (1.0 + 0.3275911 * abs_value)
            poly = 1.061405429 * reciprocal - 1.453152027
            poly = poly * reciprocal + 1.421413741
            poly = poly * reciprocal - 0.284496736
            poly = poly * reciprocal + 0.254829592
            poly = poly * reciprocal
            result = sign * (1.0 - poly * tl.exp(-abs_value * abs_value))
        else:
            # Same formula as mode 2, using exp2 to reduce transcendental overhead
            sign = tl.where(value >= 0.0, 1.0, -1.0)
            abs_value = tl.abs(value)
            reciprocal = 1.0 / (1.0 + 0.3275911 * abs_value)
            poly = 1.061405429 * reciprocal - 1.453152027
            poly = poly * reciprocal + 1.421413741
            poly = poly * reciprocal - 0.284496736
            poly = poly * reciprocal + 0.254829592
            poly = poly * reciprocal
            exponent = -abs_value * abs_value * 1.4426950408889634
            result = sign * (1.0 - poly * tl.exp2(exponent))
        tl.store(x_ptr + offsets, result, mask=mask)


def _select_block_size(num_elements, is_fp32):
    """Small shapes fall back to 1024; medium shapes use a power of two; large shapes are tiered by dtype bytes."""
    if num_elements <= 1024:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    return BLOCK_LARGE_32BIT if is_fp32 else BLOCK_LARGE_16BIT


def erf_(A):
    logger.debug("GEMS_ENFLAME ERF_")
    num_elements = A.numel()
    if num_elements == 0:
        return A
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    # Non-float / non-contiguous inputs keep the codegen path, preserving promotion and stride semantics
    if A.dtype not in supported_dtypes or not A.is_contiguous():
        return erf_func(A, out0=A)
    block_size = _select_block_size(num_elements, A.dtype == torch.float32)
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    with torch_device_fn.device(A.device):
        _erf_inplace_kernel[(grid_size,)](
            A,
            num_elements,
            BLOCK=block_size,
            MATH_MODE=MATH_MODE,
            num_warps=NUM_WARPS,
        )
    return A


__all__ = ["erf", "erf_"]
