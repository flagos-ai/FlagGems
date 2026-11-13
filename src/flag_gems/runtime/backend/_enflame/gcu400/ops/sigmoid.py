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
@triton.jit(do_not_specialize=["N_total"])
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
@triton.jit(do_not_specialize=["N_total"])
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


def _choose_block(N_total, itemsize):
    # do_bench(median) sweep, kernel mode, vs torch sigmoid on GCU400/L600:
    #   tiny shapes want the exact next_pow2 tile (fp16 N=4096 3.31us vs 3.74us @8192);
    #   2-byte large shapes peak at 131072 (fp16 536M 711us vs 777us @32768);
    #   4-byte mid shapes peak at 32768 (fp32 16.7M 48.1us vs 50.4us @65536)
    #   while 4-byte 536M prefers 65536 (1338us vs 1348us @32768).
    if N_total <= 8192:
        return max(256, triton.next_power_of_2(N_total))
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    if itemsize <= 2:
        return 131072 if N_total > (1 << 28) else 65536
    return 32768 if N_total <= (1 << 25) else 65536


def sigmoid(self):
    logger.debug("GEMS_ENFLAME SIGMOID")
    inp = self.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)
    if N_total == 0:
        return out

    BLOCK = _choose_block(N_total, inp.element_size())
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


def _choose_block_backward(N_total, itemsize):
    # backward 有 3 次访存（读 dy / 读 y / 写 dx），最优 tile 与 forward 不同：
    #   fp16: 16.7M B32768 = 44.0us（B65536 44.8us），536M B65536 = 1138.8us；
    #   fp32: 16.7M B32768 = 74.9us（B65536 75.5us，codegen 81.8us），
    #         536M B32768 = 2174.9us（B65536 2248.1us，native 2274.6us）。
    if N_total <= 8192:
        return max(256, triton.next_power_of_2(N_total))
    if N_total <= 65536:
        return 8192
    if itemsize >= 4:
        return 32768
    if N_total <= (1 << 25):
        return 32768
    return 65536


def sigmoid_backward(grad_output, output):
    logger.debug("GEMS_ENFLAME SIGMOID_BACKWARD")
    # 浮点类型用手写 flat kernel（fp32 codegen 路径在 (64,64) 上 31.9us vs flat
    # 3.7us——pointwise_dynamic Python 开销主导；fp64 / 整数、混合 dtype/device
    # 保留 codegen）。
    if (
        output.dtype != grad_output.dtype
        or output.device != grad_output.device
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return sigmoid_backward_kernel(grad_output, output)

    grad = grad_output.contiguous()
    out_y = output.contiguous()
    N_total = out_y.numel()
    if N_total == 0:
        return torch.empty_like(out_y)
    grad_input = torch.empty_like(out_y)

    BLOCK = _choose_block_backward(N_total, out_y.element_size())
    if triton.cdiv(N_total, BLOCK) * BLOCK > INT32_MAX:
        return sigmoid_backward_kernel(grad_output, output)
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

    BLOCK = _choose_block(N_total, inp.element_size())
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
