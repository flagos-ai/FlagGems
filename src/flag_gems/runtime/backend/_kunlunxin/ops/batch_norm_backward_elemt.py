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

MAX_BLOCK = 8192


def _block_size(spatial):
    block = 32
    while block < spatial and block < MAX_BLOCK:
        block *= 2
    return block


@libentry()
@triton.jit
def batch_norm_backward_elemt_kernel(
    grad_out_ptr,
    input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_dy_ptr,
    sum_dy_xmu_ptr,
    count_ptr,
    grad_input_ptr,
    spatial,
    batch_stride,
    batch,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    blk = tl.program_id(1)

    inv_count = 1.0 / tl.load(count_ptr).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    sum_dy = tl.load(sum_dy_ptr + c).to(tl.float32)
    sum_dy_xmu = tl.load(sum_dy_xmu_ptr + c).to(tl.float32)

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + c).to(tl.float32)
    else:
        weight = 1.0

    invstd_sq_sum_dy_xmu = invstd * invstd * sum_dy_xmu
    w_invstd = weight * invstd

    local = blk * BLOCK + tl.arange(0, BLOCK)
    mask = local < spatial
    base = c * spatial + local

    for n in range(0, batch):
        offsets = base + n * batch_stride
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inp = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        xmu = inp - mean
        result = w_invstd * (
            grad_out - (sum_dy + xmu * invstd_sq_sum_dy_xmu) * inv_count
        )
        tl.store(grad_input_ptr + offsets, result, mask=mask)


def batch_norm_backward_elemt(
    grad_out: Tensor,
    input: Tensor,
    mean: Tensor,
    invstd: Tensor,
    weight: Tensor,
    sum_dy: Tensor,
    sum_dy_xmu: Tensor,
    count: Tensor,
) -> Tensor:
    logger.debug("GEMS_KUNLUNXIN BATCH_NORM_BACKWARD_ELEMT")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    invstd = invstd.contiguous()
    sum_dy = sum_dy.contiguous()
    sum_dy_xmu = sum_dy_xmu.contiguous()

    C = input.shape[1]
    N = input.shape[0]
    spatial = 1
    for i in range(2, input.ndim):
        spatial *= input.shape[i]
    batch_stride = C * spatial

    has_weight = weight is not None
    if has_weight:
        weight = weight.contiguous()
    else:
        weight = torch.empty(0, dtype=input.dtype, device=input.device)

    grad_input = torch.empty_like(input)

    block = _block_size(spatial)
    nb = triton.cdiv(spatial, block)
    grid = (C, nb)

    with torch_device_fn.device(input.device):
        batch_norm_backward_elemt_kernel[grid](
            grad_out,
            input,
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count,
            grad_input,
            spatial,
            batch_stride,
            N,
            HAS_WEIGHT=has_weight,
            BLOCK=block,
        )

    return grad_input
