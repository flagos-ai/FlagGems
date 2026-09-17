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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

BLOCK_SIZE = 2048
NUM_WARPS = 8
PROGRAMS_PER_SM = 8


@libentry()
@triton.jit
def repeat_interleave_flat_kernel(
    inp,
    out,
    n_elements,
    repeats: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_id in tl.range(pid, n_blocks, num_programs):
        out_offsets = block_id * BLOCK_SIZE + offsets
        mask = out_offsets < n_elements
        inp_offsets = out_offsets // repeats
        values = tl.load(inp + inp_offsets, mask=mask)
        tl.store(out + out_offsets, values, mask=mask)


@libentry()
@triton.jit
def repeat_interleave_dim_kernel(
    inp,
    out,
    n_elements,
    inner_size,
    repeated_inner_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_id in tl.range(pid, n_blocks, num_programs):
        out_offsets = block_id * BLOCK_SIZE + offsets
        mask = out_offsets < n_elements
        inner_offsets = out_offsets % inner_size
        outer_offsets = out_offsets // repeated_inner_size
        inp_offsets = outer_offsets * inner_size + inner_offsets
        values = tl.load(inp + inp_offsets, mask=mask)
        tl.store(out + out_offsets, values, mask=mask)


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_ILUVATAR REPEAT_INTERLEAVE_SELF_INT")

    if dim is None:
        inp = inp.contiguous().flatten()
        dim = 0
    else:
        if dim < -inp.ndim or dim >= inp.ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
        if dim < 0:
            dim += inp.ndim

    inp_shape = list(inp.shape)
    output_shape = list(inp_shape)
    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
    if repeats == 0 or output.numel() == 0:
        return output

    inp = inp.contiguous()
    n_elements = output.numel()
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    num_sms = torch.cuda.get_device_properties(inp.device).multi_processor_count
    grid = (min(n_blocks, num_sms * PROGRAMS_PER_SM),)

    inner_size = 1
    for size in inp_shape[dim + 1 :]:
        inner_size *= size

    with torch_device_fn.device(inp.device):
        if inner_size == 1:
            repeat_interleave_flat_kernel[grid](
                inp,
                output,
                n_elements,
                repeats=repeats,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )
        else:
            repeat_interleave_dim_kernel[grid](
                inp,
                output,
                n_elements,
                inner_size,
                repeated_inner_size=repeats * inner_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )

    return output
