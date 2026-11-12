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
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

MAX_BLOCK = 1024


def _block_size(spatial, cap=MAX_BLOCK):
    block = 32
    while block < spatial and block < cap:
        block *= 2
    return block


@libentry()
@triton.jit
def batch_norm_backward_reduce_channel_kernel(
    grad_out_ptr,
    input_ptr,
    mean_ptr,
    invstd_ptr,
    sum_dy_ptr,
    sum_dy_xmu_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    spatial,
    batch_stride,
    total,
    COMPUTE_INPUT_G: tl.constexpr,
    COMPUTE_GRAD_WEIGHT: tl.constexpr,
    COMPUTE_GRAD_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)

    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    if COMPUTE_INPUT_G or COMPUTE_GRAD_BIAS:
        acc_sum_dy = tl.zeros([BLOCK], tl.float32)
    if COMPUTE_INPUT_G:
        acc_sum_dy_xmu = tl.zeros([BLOCK], tl.float32)
    if COMPUTE_GRAD_WEIGHT:
        acc_grad_weight = tl.zeros([BLOCK], tl.float32)

    for start in range(0, total, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < total
        offsets = (idx // spatial) * batch_stride + c * spatial + (idx % spatial)
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inp = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        xmu = inp - mean
        if COMPUTE_INPUT_G or COMPUTE_GRAD_BIAS:
            acc_sum_dy += grad_out
        if COMPUTE_INPUT_G:
            acc_sum_dy_xmu += grad_out * xmu
        if COMPUTE_GRAD_WEIGHT:
            acc_grad_weight += grad_out * xmu * invstd

    if COMPUTE_INPUT_G:
        tl.store(sum_dy_ptr + c, tl.sum(acc_sum_dy))
        tl.store(sum_dy_xmu_ptr + c, tl.sum(acc_sum_dy_xmu))
    if COMPUTE_GRAD_WEIGHT:
        tl.store(grad_weight_ptr + c, tl.sum(acc_grad_weight))
    if COMPUTE_GRAD_BIAS:
        tl.store(grad_bias_ptr + c, tl.sum(acc_sum_dy))


def batch_norm_backward_reduce(
    grad_out: Tensor,
    input: Tensor,
    mean: Tensor,
    invstd: Tensor,
    weight: Tensor,
    input_g: bool,
    weight_g: bool,
    bias_g: bool,
) -> tuple:
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_BACKWARD_REDUCE")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    invstd = invstd.contiguous()

    C = input.shape[1]
    N = input.shape[0]
    spatial = input.numel() // (N * C)
    batch_stride = C * spatial
    elements_per_channel = N * spatial

    device = grad_out.device
    sum_dy = (
        torch.empty(C, dtype=torch.float32, device=device)
        if input_g
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    sum_dy_xmu = (
        torch.empty(C, dtype=torch.float32, device=device)
        if input_g
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    grad_weight = (
        torch.empty(C, dtype=grad_out.dtype, device=device)
        if weight_g
        else torch.empty(0, dtype=grad_out.dtype, device=device)
    )
    grad_bias = (
        torch.empty(C, dtype=grad_out.dtype, device=device)
        if bias_g
        else torch.empty(0, dtype=grad_out.dtype, device=device)
    )

    with torch_device_fn.device(device):
        batch_norm_backward_reduce_channel_kernel[(C,)](
            grad_out,
            input,
            mean,
            invstd,
            sum_dy,
            sum_dy_xmu,
            grad_weight,
            grad_bias,
            spatial,
            batch_stride,
            elements_per_channel,
            COMPUTE_INPUT_G=input_g,
            COMPUTE_GRAD_WEIGHT=weight_g,
            COMPUTE_GRAD_BIAS=bias_g,
            BLOCK=_block_size(spatial),
        )

    return (
        sum_dy if input_g else None,
        sum_dy_xmu if input_g else None,
        grad_weight if weight_g else None,
        grad_bias if bias_g else None,
    )
