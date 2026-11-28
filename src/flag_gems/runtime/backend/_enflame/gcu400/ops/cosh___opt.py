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

# Number of resident CTAs aligned with the GCU400 SIP topology; reused inside the kernel via a strided loop
NUM_CTAS = 48


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _cosh_inplace_kernel(x_ptr, num_elements, BLOCK: tl.constexpr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_tiles = (num_elements + BLOCK - 1) // BLOCK
    for tile_index in tl.range(program_id, num_tiles, num_programs):
        offsets = tile_index * BLOCK + arange
        mask = offsets < num_elements
        # fp32 intermediate computation preserves precision for low-precision inputs; this body is at the floor under
        # fp32 bandwidth
        value = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        result = 0.5 * (tl.exp(value) + tl.exp(-value))
        tl.store(x_ptr + offsets, result.to(x_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _cosh_inplace_single_exp_kernel(x_ptr, num_elements, BLOCK: tl.constexpr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_tiles = (num_elements + BLOCK - 1) // BLOCK
    for tile_index in tl.range(program_id, num_tiles, num_programs):
        offsets = tile_index * BLOCK + arange
        mask = offsets < num_elements
        value = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        # cosh(x) = 0.5*(exp(x) + 1/exp(x)); 1/t is implemented via the rsqrt(t)^rsqrt(t) identity
        # lowered to hardware primitives, eliminating the second transcendental evaluation and IEEE division slow paths
        # in the double-exp body
        t = tl.exp(value)
        reciprocal_sqrt = tl.rsqrt(t)
        result = 0.5 * (t + reciprocal_sqrt * reciprocal_sqrt)
        tl.store(x_ptr + offsets, result.to(x_ptr.dtype.element_ty), mask=mask)


def _select_block_size(num_elements, is_fp32):
    """Power-of-two tiles for small shapes, dtype tiers for large ones (measured sweet spots)."""
    if num_elements <= 1024:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    if is_fp32:
        # At 16.7M, 32768/nw4 is best (48.9 vs 50.5us @65536);
        # At 512M, 65536/nw4 is best (1341 vs 1375us @32768)
        if num_elements <= (1 << 24):
            return 32768
        return 65536
    # 16-bit: mid-size 65536/nw2 (16.7M 38.8 vs 40.1us @32768);
    # For >2^28, 131072/nw2 (512M 921 vs 967us @65536)
    if num_elements <= (1 << 19):
        return 32768
    if num_elements <= (1 << 28):
        return 65536
    return 131072


def _launch_cosh_inplace(A, num_elements, is_fp32):
    block_size = _select_block_size(num_elements, is_fp32)
    num_warps = 4 if is_fp32 else 2
    kernel = _cosh_inplace_kernel if is_fp32 else _cosh_inplace_single_exp_kernel
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    # Occasional cold-compilation failures in the external llvm converter (P02 toolchain with converterResult=-2
    # flake family); 3 launch retries ride out the transient window, and deterministic corruption still raises on the
    # 3rd attempt
    # (precedents in the same family: sinh.py / acosh.py / atan2_.py)
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                kernel[(grid_size,)](
                    A,
                    num_elements,
                    BLOCK=block_size,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "cosh_ flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def cosh_(A):
    logger.debug("GEMS_ENFLAME COSH_")
    num_elements = A.numel()
    if num_elements == 0:
        return A
    # Non-floating-point / non-contiguous inputs use the common codegen path to preserve promotion and stride semantics
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if A.dtype not in supported_dtypes or not A.is_contiguous():
        return _common_cosh_func(A, out0=A)
    _launch_cosh_inplace(A, num_elements, A.dtype == torch.float32)
    return A


__all__ = ["cosh_"]
