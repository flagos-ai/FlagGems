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


# tanh via 2*sigmoid(2x)-1: tl.sigmoid lowers to a native sigmoid instruction
# while the previous (exp(2x)-1)/(exp(2x)+1) form pays a full fp32 division in
# the hot loop (536M fp16: 1950.7us -> 931.9us at identical config 65536/nw4/g96,
# -> 833us at the final nw2 config; do_bench kernel mode, GCU400/L600).  The
# sigmoid form is also overflow-free for |x| > 44 where exp(2x) saturates to
# inf and (inf-1)/(inf+1) yields NaN.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
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
        out = 2.0 * tl.sigmoid(2.0 * x) - 1.0
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


def _choose_block(N_total, itemsize):
    # do_bench(median) sweep, kernel mode, tanh = 2*sigmoid(2x)-1, GCU400/L600:
    #   tiny shapes want the exact next_pow2 tile (fp16 N=4096 3.47us vs 4.37 @8192);
    #   2-byte large shapes peak at 131072 (fp16 536M 821-847us vs 833-844 @65536);
    #   4-byte mid shapes peak at 32768 (fp32 16.7M 49.1us vs 49.3us @65536)
    #   while 4-byte 536M prefers 65536 (1359us vs 1413 @131072).
    if N_total <= 8192:
        return max(256, triton.next_power_of_2(N_total))
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    if itemsize <= 2:
        return 131072 if N_total > (1 << 28) else 65536
    return 32768 if N_total <= (1 << 25) else 65536


def tanh(self):
    logger.debug("GEMS_ENFLAME TANH")
    inp = self.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)
    if N_total == 0:
        return out

    BLOCK = _choose_block(N_total, inp.element_size())
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    with torch_device_fn.device(inp.device):
        tanh_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=2,
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
    # Empty tensors have no launchable CTA; return early to avoid a 0-CTA grid.
    if N_total == 0:
        return A
    # The kernel writes a contiguous buffer by linear offset; the contiguous
    # copy of a non-contiguous input has a different layout, so write a temp
    # buffer and copy back along the original strides to keep in-place
    # semantics (same pattern as sigmoid_).
    output = A if inp.data_ptr() == A.data_ptr() else torch.empty_like(inp)

    BLOCK = _choose_block(N_total, inp.element_size())
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    with torch_device_fn.device(inp.device):
        tanh_flat_kernel[(grid_size,)](
            inp,
            output,
            N_total,
            BLOCK=BLOCK,
            num_warps=2,
        )
    if output is not A:
        A.copy_(output)

    return A
