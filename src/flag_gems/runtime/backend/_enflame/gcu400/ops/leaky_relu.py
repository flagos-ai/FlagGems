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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
BLOCK = 8192


@libentry()
@triton.jit(do_not_specialize=["N_total", "negative_slope"])
def _leaky_relu_kernel(
    input_ptr,
    output_ptr,
    N_total,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        x = tl.load(input_ptr + off, mask=mask).to(tl.float32)
        output = tl.where(x >= 0, x, x * negative_slope)
        tl.store(output_ptr + off, output, mask=mask)


def _grid(n_elements):
    return min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)


def _launch_leaky_relu(input_ptr, output_ptr, n_elements, negative_slope):
    if n_elements == 0:
        return
    grid = _grid(n_elements)
    with torch_device_fn.device(input_ptr.device):
        _leaky_relu_kernel[(grid,)](
            input_ptr,
            output_ptr,
            n_elements,
            negative_slope,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_ENFLAME LEAKY_RELU")
    if not A.is_contiguous():
        A = A.contiguous()
    output = torch.empty_like(A)
    _launch_leaky_relu(A, output, A.numel(), negative_slope)
    return output


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_")
    if not A.is_contiguous():
        raise RuntimeError(
            "leaky_relu_ requires a contiguous tensor for in-place operation"
        )
    _launch_leaky_relu(A, A, A.numel(), negative_slope)
    return A


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu(A, negative_slope)
    if not A.is_contiguous():
        A = A.contiguous()
    _launch_leaky_relu(A, out, A.numel(), negative_slope)
    return out
