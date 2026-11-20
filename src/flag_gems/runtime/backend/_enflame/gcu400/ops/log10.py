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

"""
GCU400-specific log10: 1D flat Triton kernel + fixed-size grid, replacing pointwise_dynamic codegen.

This variant is rendered by the optimization iterations: math mode, BLOCK tiering, grid size, num_warps,
loop software pipelining and memory-alignment hints are all scannable launch-time parameters.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Fixed program count matches GCU400 compute units; an in-kernel stride loop covers all tiles.
NUM_PROGRAMS = 48
BLOCK_SIZE = 16384
BLOCK_SMALL = 4096
NUM_WARPS = 2


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def _log10_flat_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK: tl.constexpr,
    MATH_MODE: tl.constexpr,
    ALIGN_HINT: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(program_id, tl.cdiv(num_elements, BLOCK), num_programs):
        base_offset = tl.multiple_of(block_id * BLOCK, BLOCK)
        element_offsets = base_offset + tl.arange(0, BLOCK)
        valid_mask = element_offsets < num_elements
        # Integral / bool inputs are computed in fp32; output dtype is decided by dispatch.
        input_values = tl.load(
            input_ptr + element_offsets, mask=valid_mask, other=0
        ).to(tl.float32)
        if MATH_MODE == 0:
            # Multiply by 1/ln(10): the current engineering baseline implementation.
            result_values = tl.log(input_values) * 0.4342944819032518
        elif MATH_MODE == 1:
            # Divide by ln(10): avoids double rounding from a reciprocal constant, fixing an fp32 1ulp difference.
            result_values = tl.log(input_values) / 2.302585092994046
        elif MATH_MODE == 2:
            # log2 times log10(2): tests whether the backend has a faster log2 hardware path.
            result_values = tl.log2(input_values) * 0.3010299956639812
        else:
            result_values = tl.log2(input_values) / 3.321928094887362
        tl.store(output_ptr + element_offsets, result_values, mask=valid_mask)


def _output_dtype(input_tensor: torch.Tensor):
    # Consistent with the common implementation's INT_TO_FLOAT promotion rules.
    if input_tensor.is_floating_point():
        return input_tensor.dtype
    return torch.float32


def _launch_log10(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    num_elements = input_tensor.numel()
    if num_elements == 0:
        return output_tensor
    block_size = (
        BLOCK_SMALL
        if num_elements <= 1048576
        else (BLOCK_SIZE * 2 if num_elements >= 67108864 else BLOCK_SIZE)
    )
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_PROGRAMS)
    with torch_device_fn.device(input_tensor.device):
        _log10_flat_kernel[(grid_size,)](
            input_tensor,
            output_tensor,
            num_elements,
            BLOCK=block_size,
            MATH_MODE=0,
            ALIGN_HINT=True,
            num_warps=NUM_WARPS,
        )
    return output_tensor


def log10(A):
    logger.debug("GEMS_ENFLAME LOG10")
    contiguous_input = A if A.is_contiguous() else A.contiguous()
    output = torch.empty(
        contiguous_input.shape, dtype=_output_dtype(A), device=A.device
    )
    return _launch_log10(contiguous_input, output)


def log10_(A):
    logger.debug("GEMS_ENFLAME LOG10_")
    if A.is_contiguous():
        return _launch_log10(A, A)
    # Non-contiguous in-place inputs compute on a linear copy first, then write back by the original stride.
    contiguous_copy = A.contiguous()
    _launch_log10(contiguous_copy, contiguous_copy)
    A.copy_(contiguous_copy)
    return A


def log10_out(A, out):
    logger.debug("GEMS_ENFLAME LOG10_OUT")
    contiguous_input = A if A.is_contiguous() else A.contiguous()
    if out.is_contiguous():
        return _launch_log10(contiguous_input, out)
    contiguous_output = torch.empty(out.shape, dtype=out.dtype, device=out.device)
    _launch_log10(contiguous_input, contiguous_output)
    out.copy_(contiguous_output)
    return out
