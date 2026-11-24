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
# Pipeline depth of the grid-stride loop: keep 3 as "harmless and potentially beneficial"; measurements and
# counterexamples are in the module
# docstring (for in-place and 16-bit, st=1/3 makes no difference; only in the two-pass scans of fp32 512M
# is st>=2 faster). It must be a tl.constexpr to be read by tl.range inside @triton.jit.
GRID_STRIDE_STAGES = tl.constexpr(3)


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def atanh_kernel(
    x_ptr,
    out_ptr,
    num_elements,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = (num_elements + BLOCK - 1) // BLOCK
    for tile_index in tl.range(program_id, num_tiles, num_programs, num_stages=1):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        value = x.to(tl.float32)
        # sqrt((1+v)/(1-v)) = (1+v) * rsqrt(1-v) * rsqrt(1-v)
        reciprocal_sqrt = tl.rsqrt(1.0 - value)
        scaled = ((1.0 + value) * reciprocal_sqrt) * reciprocal_sqrt
        log_value = tl.log(scaled)
        tl.store(out_ptr + offsets, (log_value.to(x.dtype)) * 0.5, mask=mask)


def _choose_block(num_elements, is_fp32):
    """1024 for small shapes, the nearest power of two for medium ones, dtype tiers for large ones."""
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
            num_warps=NUM_WARPS,
        )
    return out


def _atanh(A):
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def atanh(A):
    logger.debug("GEMS_ENFLAME ATANH")
    # The flat kernel allocates with empty_like and stores back in the input
    # dtype, which only matches ATen for float inputs.  ATen promotes integer
    # inputs to fp32, so leave those to the generic INT_TO_FLOAT pointwise op.
    if A.dtype not in _FLOAT_DTYPES:
        from flag_gems.ops.atanh import atanh as _generic_atanh

        return _generic_atanh(A)
    return _atanh(A)


def atanh_(A):
    logger.debug("GEMS_ENFLAME ATANH_")
    # ATen rejects in-place atanh on integer/bool inputs (the float result
    # cannot be cast back into the output buffer), matching the generic op.
    if not A.dtype.is_floating_point:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type " f"{A.dtype}"
        )
    if A.dtype not in _FLOAT_DTYPES:
        # fp64 (and any non-flat-kernel float dtype) goes through the generic
        # in-place op; raising here would wrongly reject valid float inputs.
        from flag_gems.ops.atanh import atanh_ as _generic_atanh_

        return _generic_atanh_(A)
    if not A.is_contiguous():
        A.copy_(_atanh(A))
        return A
    return _launch(A, A)
