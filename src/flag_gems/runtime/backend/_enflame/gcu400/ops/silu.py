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
from flag_gems.utils.triton_lang_extension import div_rn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit
def silu_flat_kernel(
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
        sigma = 1.0 / (1.0 + tl.exp(-x))
        out = x * sigma
        tl.store(out_ptr + off, out, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_backward_kernel(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


# Direct-pointer flat kernel, same shape as silu_flat_kernel.  The
# pointwise_dynamic variant is much slower on gcu400: div_rn lowers to a
# slow scalar libdevice call, and the block-pointer form adds overhead.
# A plain division is fast and matches torch bit-for-bit for fp16 here.
@libentry()
@triton.jit
def silu_backward_flat_kernel(
    x_ptr,
    dy_ptr,
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
        dy = tl.load(dy_ptr + off, mask=mask).to(tl.float32)
        sigma = 1.0 / (1.0 + tl.exp(-x))
        dx = dy * sigma * (1.0 + x * (1.0 - sigma))
        tl.store(out_ptr + off, dx, mask=mask)


def _choose_block(N_total):
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    if N_total > (1 << 28):
        # 超大 shape 上 65536 元素/tile 的寄存器压力导致明显劣化，
        # 实测 536M 元素 fp32 1633us -> 1412us、fp16 1489us -> 929us
        return 32768
    return 65536


def silu(self):
    logger.debug("GEMS_ENFLAME SILU")
    inp = self.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
    nw = 2

    with torch_device_fn.device(inp.device):
        silu_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return out


def silu_backward(grad_output, self):
    logger.debug("GEMS_ENFLAME SILU_BACKWARD")
    # Mixed dtype/device needs the generic promotion logic; everything else
    # goes through the flat kernel.
    if self.dtype != grad_output.dtype or self.device != grad_output.device:
        return silu_backward_kernel(self, grad_output)

    inp = self.contiguous()
    grad = grad_output.contiguous()
    N_total = inp.numel()
    if N_total == 0:
        return torch.empty_like(inp)
    out = torch.empty_like(inp)

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
    with torch_device_fn.device(inp.device):
        silu_backward_flat_kernel[(grid_size,)](
            inp,
            grad,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=2,
        )

    return out


def silu_(A):
    logger.debug("GEMS_ENFLAME SILU_")
    inp = A.contiguous()
    N_total = inp.numel()

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
    nw = 2

    with torch_device_fn.device(inp.device):
        silu_flat_kernel[(grid_size,)](
            inp,
            A,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return A
