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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit
def sigmoid_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + off, out, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward_kernel(dy, y):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


def _choose_block(N_total):
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    return 65536


def sigmoid(self):
    logger.debug("GEMS_ENFLAME SIGMOID")
    inp = self.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 4)

    nw = 2 if N_total <= 134217728 else 4

    with torch_device_fn.device(inp.device):
        sigmoid_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return out


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS_ENFLAME SIGMOID_BACKWARD")
    grad_input = sigmoid_backward_kernel(grad_output, output)
    return grad_input


def sigmoid_(A):
    logger.debug("GEMS_ENFLAME SIGMOID_")
    inp = A.contiguous()
    N_total = inp.numel()

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 4)

    nw = 2 if N_total <= 134217728 else 4

    with torch_device_fn.device(inp.device):
        sigmoid_flat_kernel[(grid_size,)](
            inp,
            A,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return A
