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


# Direct-pointer flat kernel, same shape as sigmoid_flat_kernel.  On gcu400 the
# pointwise_dynamic wrapper adds about 100us of Python-side overhead to every
# call, which dominates all core shapes except the 536M-element one; a plain
# grid-stride kernel removes it and keeps the same fp32 accumulation.
@libentry()
@triton.jit
def sigmoid_backward_flat_kernel(
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
        dx = dy * (1.0 - y) * y
        tl.store(out_ptr + off, dx, mask=mask)


def _choose_block(N_total):
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    if N_total > (1 << 28):
        # 超大 shape 上 65536 元素/tile 的寄存器压力导致明显劣化
        return 32768
    return 65536


def sigmoid(self):
    logger.debug("GEMS_ENFLAME SIGMOID")
    inp = self.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    nw = 2

    with torch_device_fn.device(inp.device):
        sigmoid_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return out


def _choose_block_backward(N_total):
    # backward 有 3 次访存（读 dy / 读 y / 写 dx），最优 tile 与 forward 不同：
    # 16.7M 元素 B32768 = 44.0us（B65536 44.8us），536M 元素 B65536 = 1138.8us
    if N_total <= 65536:
        return 8192
    if N_total <= (1 << 25):
        return 32768
    return 65536


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS_ENFLAME SIGMOID_BACKWARD")
    # fp16/bf16 用手写 flat kernel；fp32 / fp64 / 整数、混合 dtype/device 保留
    # codegen（fp32 中、大 shape 上 3 轮一致显示手写路径更慢）。
    if (
        output.dtype != grad_output.dtype
        or output.device != grad_output.device
        or output.dtype not in (torch.float16, torch.bfloat16)
    ):
        return sigmoid_backward_kernel(grad_output, output)

    grad = grad_output.contiguous()
    out_y = output.contiguous()
    N_total = out_y.numel()
    if N_total == 0:
        return torch.empty_like(out_y)
    grad_input = torch.empty_like(out_y)

    BLOCK = _choose_block_backward(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
    with torch_device_fn.device(out_y.device):
        sigmoid_backward_flat_kernel[(grid_size,)](
            grad,
            out_y,
            grad_input,
            N_total,
            BLOCK=BLOCK,
            num_warps=2,
        )
    return grad_input


def sigmoid_(A):
    logger.debug("GEMS_ENFLAME SIGMOID_")
    inp = A.contiguous()
    N_total = inp.numel()

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    nw = 2

    with torch_device_fn.device(inp.device):
        sigmoid_flat_kernel[(grid_size,)](
            inp,
            A,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return A
