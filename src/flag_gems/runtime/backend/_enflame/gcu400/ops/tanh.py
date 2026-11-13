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
INT32_MAX = torch.iinfo(torch.int32).max


@libentry()
@triton.jit
def tanh_flat_kernel(
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
        e2x = tl.exp(2.0 * x)
        out = (e2x - 1.0) / (e2x + 1.0)
        tl.store(out_ptr + off, out, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_kernel(y, dy):
    y = y.to(tl.float32)
    return dy.to(tl.float32) * (1.0 - y * y)


# Direct-pointer flat kernel, same shape as sigmoid_backward_flat_kernel (the
# production precedent).  On gcu400 the pointwise_dynamic wrapper costs tens of
# us of Python-side overhead per call through the aten dispatch path, which
# dominates every core shape except the 536M-element ones, and its generated
# bptr kernel runs ~10-20% behind the flat grid-stride form on mid shapes.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def tanh_backward_flat_kernel(
    dy_ptr,
    y_ptr,
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
        dy = tl.load(dy_ptr + off, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + off, mask=mask).to(tl.float32)
        dx = dy * (1.0 - y * y)
        tl.store(out_ptr + off, dx, mask=mask)


def _choose_block_backward(N_total, itemsize):
    # do_bench(median) sweep, kernel mode, vs torch tanh_backward on GCU400/L600
    # (3 memory streams: read dy / read y / write dx, so tile ladder differs
    # from the 2-stream forward):
    #   fp16: 16.7M B32768 = 43.9us (B16384 50.2, B65536 45.3), 536M B65536
    #         nw1 = 1.151ms (B32768 1.176, B131072 1.38 = DSM spill);
    #   fp32: 16.7M B16384/B32768 = 78.2/78.8us (B65536 87.0), 536M B32768
    #         nw1 = 2.265ms (B65536 2.46+);
    #   tiny shapes want the exact next_pow2 tile ((64,64) 3.3us flat direct).
    if N_total <= 8192:
        return max(256, triton.next_power_of_2(N_total))
    if N_total <= 65536:
        return 8192
    if itemsize >= 4:
        return 32768
    if N_total <= (1 << 25):
        return 32768
    return 65536


def _choose_block(N_total):
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    return 65536


def tanh(self):
    logger.debug("GEMS_ENFLAME TANH")
    inp = self.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 4)

    nw = 2 if N_total <= 134217728 else 4

    with torch_device_fn.device(inp.device):
        tanh_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return out


def tanh_backward(grad_output, output):
    logger.debug("GEMS_ENFLAME TANH_BACKWARD")
    # Floating-point inputs go through the hand-written flat kernel; mixed
    # dtype/device, fp64/int and int32-overflowing sizes keep the codegen path.
    if (
        output.dtype != grad_output.dtype
        or output.device != grad_output.device
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return tanh_backward_kernel(output, grad_output)

    grad = grad_output.contiguous()
    out_y = output.contiguous()
    N_total = out_y.numel()
    if N_total == 0:
        return torch.empty_like(out_y)

    BLOCK = _choose_block_backward(N_total, out_y.element_size())
    if triton.cdiv(N_total, BLOCK) * BLOCK > INT32_MAX:
        return tanh_backward_kernel(output, grad_output)
    grad_input = torch.empty_like(out_y)

    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
    nw = 1 if N_total > (1 << 25) else 2

    with torch_device_fn.device(out_y.device):
        tanh_backward_flat_kernel[(grid_size,)](
            grad,
            out_y,
            grad_input,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )
    return grad_input


def tanh_(A):
    logger.debug("GEMS_ENFLAME TANH_")
    inp = A.contiguous()
    N_total = inp.numel()

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 4)

    nw = 2 if N_total <= 134217728 else 4

    with torch_device_fn.device(inp.device):
        tanh_flat_kernel[(grid_size,)](
            inp,
            A,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return A
