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
@triton.jit(do_not_specialize=["N_total", "lower", "upper", "training"])
def rrelu_with_noise_backward_kernel(
    grad_out_ptr,
    input_ptr,
    noise_ptr,
    grad_in_ptr,
    N_total,
    lower,
    upper,
    training,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    slope = (lower + upper) * 0.5
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        mask = off < N_total
        go = tl.load(grad_out_ptr + off, mask=mask, other=0)
        x = tl.load(input_ptr + off, mask=mask, other=0)
        nz = tl.load(noise_ptr + off, mask=mask, other=0)
        go_f32 = go.to(tl.float32)
        x_f32 = x.to(tl.float32)
        nz_f32 = nz.to(tl.float32)
        grad_train = go_f32 * nz_f32
        grad_eval = go_f32 * tl.where(x_f32 > 0, 1.0, slope)
        cond = tl.full(go_f32.shape, training, tl.int1)
        grad_f32 = tl.where(cond, grad_train, grad_eval)
        tl.store(grad_in_ptr + off, grad_f32.to(go.dtype), mask=mask)


def _launch_rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    out: torch.Tensor,
):
    go = grad_output.contiguous()
    x = input.contiguous()
    nz = noise.contiguous()
    out_t = out.contiguous()
    n_elements = out_t.numel()
    if n_elements == 0:
        return out
    grid = min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(grad_output.device):
        rrelu_with_noise_backward_kernel[(grid,)](
            go,
            x,
            nz,
            out_t,
            n_elements,
            float(lower),
            float(upper),
            1 if training else 0,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )
    if out is not out_t:
        out.copy_(out_t)
    return out


def rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool = False,
):
    logger.debug("GEMS_ENFLAME RRELU_WITH_NOISE_BACKWARD")
    out = torch.empty_like(grad_output)
    return _launch_rrelu_with_noise_backward(
        grad_output, input, noise, lower, upper, training, out
    )
