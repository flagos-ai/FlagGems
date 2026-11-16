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

# XPU3: tl.sum over an unmasked 1-D tile is correct up to 8192 lanes; larger
# tiles need an explicit buffer_size_limit and are not worth it here
# (see KernelOpt_Knowledge wiki technique-xpu3-flat-1d-constexpr-rowwidth).
MIN_BLOCK = 1024
MAX_BLOCK = 8192


def _pick_block(total):
    """Largest power-of-two tile in [MIN_BLOCK, MAX_BLOCK] covering ``total``.

    A single (masked) tile is used for small channels; large channels stream in
    ``MAX_BLOCK``-lane chunks.  ``spatial`` is passed as a ``tl.constexpr`` so
    the ``// spatial`` / ``% spatial`` address decomposition lowers to constant
    magic-multiply instead of the ~30-instruction runtime soft-division
    sequence, which was the dominant cost of the previous implementation.
    """
    block = MIN_BLOCK
    while block < total and block < MAX_BLOCK:
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
    batch_stride,
    total,
    spatial: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    COMPUTE_INPUT_G: tl.constexpr,
    COMPUTE_GRAD_WEIGHT: tl.constexpr,
    COMPUTE_GRAD_BIAS: tl.constexpr,
):
    NEED_SUM_DY: tl.constexpr = COMPUTE_INPUT_G or COMPUTE_GRAD_BIAS
    NEED_SUM_DY_XMU: tl.constexpr = COMPUTE_INPUT_G or COMPUTE_GRAD_WEIGHT

    c = tl.program_id(0)

    mean = tl.load(mean_ptr + c).to(tl.float32)

    if NEED_SUM_DY:
        acc_sum_dy = tl.zeros([BLOCK], tl.float32)
    if NEED_SUM_DY_XMU:
        acc_sum_dy_xmu = tl.zeros([BLOCK], tl.float32)

    base = c * spatial
    for start in range(0, total, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        if EVEN:
            offsets = base + (idx // spatial) * batch_stride + (idx % spatial)
            grad_out = tl.load(grad_out_ptr + offsets).to(tl.float32)
            if NEED_SUM_DY_XMU:
                inp = tl.load(input_ptr + offsets).to(tl.float32)
        else:
            mask = idx < total
            offsets = base + (idx // spatial) * batch_stride + (idx % spatial)
            grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            if NEED_SUM_DY_XMU:
                inp = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if NEED_SUM_DY:
            acc_sum_dy += grad_out
        if NEED_SUM_DY_XMU:
            acc_sum_dy_xmu += grad_out * (inp - mean)

    if NEED_SUM_DY:
        sdy = tl.sum(acc_sum_dy)
    if NEED_SUM_DY_XMU:
        sdyx = tl.sum(acc_sum_dy_xmu)

    if COMPUTE_INPUT_G:
        tl.store(sum_dy_ptr + c, sdy)
        tl.store(sum_dy_xmu_ptr + c, sdyx)
    if COMPUTE_GRAD_WEIGHT:
        # grad_weight_c = invstd_c * sum(dy * (x - mean_c)); invstd is a
        # per-channel scalar so the multiply can be hoisted out of the reduce.
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        tl.store(
            grad_weight_ptr + c, (sdyx * invstd).to(grad_weight_ptr.dtype.element_ty)
        )
    if COMPUTE_GRAD_BIAS:
        tl.store(grad_bias_ptr + c, sdy.to(grad_bias_ptr.dtype.element_ty))


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
    device = grad_out.device
    dt = grad_out.dtype

    # Outputs: sum_dy / sum_dy_xmu are always fp32; grad_weight / grad_bias
    # follow the input dtype.
    sum_dy = torch.empty(C, dtype=torch.float32, device=device) if input_g else None
    sum_dy_xmu = torch.empty(C, dtype=torch.float32, device=device) if input_g else None
    grad_weight = torch.empty(C, dtype=dt, device=device) if weight_g else None
    grad_bias = torch.empty(C, dtype=dt, device=device) if bias_g else None

    if not (input_g or weight_g or bias_g):
        return (None, None, None, None)

    # Empty reduction domain: per-channel sums over zero elements are zero.
    # torch.zeros is used only to materialise the (well-defined) zero result,
    # not to perform any of the measured reductions.
    if input.numel() == 0 or C == 0:
        if input_g:
            sum_dy = torch.zeros(C, dtype=torch.float32, device=device)
            sum_dy_xmu = torch.zeros(C, dtype=torch.float32, device=device)
        if weight_g:
            grad_weight = torch.zeros(C, dtype=dt, device=device)
        if bias_g:
            grad_bias = torch.zeros(C, dtype=dt, device=device)
        return (sum_dy, sum_dy_xmu, grad_weight, grad_bias)

    spatial = input.numel() // (N * C)
    batch_stride = C * spatial
    total = N * spatial

    block = _pick_block(total)

    # Dummy typed pointers for outputs that are not requested (never stored to,
    # guarded by constexpr flags in the kernel).
    dy_ptr = (
        sum_dy
        if sum_dy is not None
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    dyx_ptr = (
        sum_dy_xmu
        if sum_dy_xmu is not None
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    dw_ptr = (
        grad_weight
        if grad_weight is not None
        else torch.empty(0, dtype=dt, device=device)
    )
    db_ptr = (
        grad_bias if grad_bias is not None else torch.empty(0, dtype=dt, device=device)
    )

    with torch_device_fn.device(device):
        batch_norm_backward_reduce_channel_kernel[(C,)](
            grad_out,
            input,
            mean,
            invstd,
            dy_ptr,
            dyx_ptr,
            dw_ptr,
            db_ptr,
            batch_stride,
            total,
            spatial=spatial,
            BLOCK=block,
            EVEN=(total % block) == 0,
            COMPUTE_INPUT_G=input_g,
            COMPUTE_GRAD_WEIGHT=weight_g,
            COMPUTE_GRAD_BIAS=bias_g,
        )

    return (sum_dy, sum_dy_xmu, grad_weight, grad_bias)
