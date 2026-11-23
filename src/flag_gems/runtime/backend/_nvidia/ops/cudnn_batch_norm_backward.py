# Copyright 2026, The FlagOS Contributors.
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
#
import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.ops.cudnn_batch_norm_backward import _validate_backward
from flag_gems.ops.cudnn_batch_norm_backward import (
    cudnn_batch_norm_backward as _generic_backward,
)
from flag_gems.ops.cudnn_batch_norm_backward import make_3d_for_bn
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _backward_fused(
    X,
    DY,
    W,
    Mean,
    Inv,
    DX,
    DW,
    DB,
    M: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.arange(0, B)
    offsets = r // S * C * S + c * S + r % S
    mean = tl.load(Mean + c)
    inv = tl.load(Inv + c)
    x = tl.load(X + offsets, r < M, 0)
    dy = tl.load(DY + offsets, r < M, 0)
    centered = tl.where(r < M, x - mean, 0)
    sum_dy = tl.sum(dy, 0)
    sum_xdy = tl.sum(centered * dy, 0)
    dw = sum_xdy * inv
    tl.store(DW + c, dw)
    tl.store(DB + c, sum_dy)
    a = inv * tl.load(W + c)
    b = dw * inv / M
    d = sum_dy / M
    tl.store(DX + offsets, a * (dy - (centered * b + d)), r < M)


@libentry()
@triton.jit
def _backward_partial(
    X,
    DY,
    Mean,
    Partial,
    M: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    B: tl.constexpr,
):
    c = tl.program_id(0)
    p = tl.program_id(1)
    r = p * B + tl.arange(0, B)
    offsets = r // S * C * S + c * S + r % S
    x = tl.load(X + offsets, r < M, 0)
    dy = tl.load(DY + offsets, r < M, 0)
    # Apply invstd after merging, rather than multiplying every input by it.
    centered = tl.where(r < M, x - tl.load(Mean + c), 0)
    tl.store(Partial + c * P * 2 + p, tl.sum(dy, 0))
    tl.store(Partial + c * P * 2 + P + p, tl.sum(centered * dy, 0))


@libentry()
@triton.jit
def _backward_apply(
    X,
    DY,
    W,
    Mean,
    Inv,
    Partial,
    DX,
    DW,
    DB,
    M: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    PB: tl.constexpr,
    B: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    # Each output CTA merges the small partial array; no serial output pass.
    p = tl.arange(0, PB)
    sum_dy = tl.sum(tl.load(Partial + c * P * 2 + p, p < P, 0), 0)
    sum_xdy = tl.sum(tl.load(Partial + c * P * 2 + P + p, p < P, 0), 0)
    inv = tl.load(Inv + c)
    dw = sum_xdy * inv
    if block == 0:
        tl.store(DW + c, dw)
        tl.store(DB + c, sum_dy)
    a = inv * tl.load(W + c)
    b = dw * inv / M
    d = sum_dy / M
    mean = tl.load(Mean + c)
    r = block * B + tl.arange(0, B)
    offsets = r // S * C * S + c * S + r % S
    x = tl.load(X + offsets, r < M, 0)
    dy = tl.load(DY + offsets, r < M, 0)
    tl.store(DX + offsets, a * (dy - ((x - mean) * b + d)), r < M)


def cudnn_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Tensor = None,
    running_var: Tensor = None,
    save_mean: Tensor = None,
    save_var: Tensor = None,
    epsilon: float = 1e-5,
    reserveSpace: Tensor = None,
):
    logger.debug("GEMS_NVIDIA CUDNN_BATCH_NORM_BACKWARD")
    if not (
        input.dtype == torch.float64
        and input.is_contiguous()
        and grad_output.is_contiguous()
    ):
        return _generic_backward(
            input,
            grad_output,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_var,
            epsilon,
            reserveSpace,
        )
    _validate_backward(input, grad_output, weight, save_mean, save_var)
    input_3d = make_3d_for_bn(input)
    grad_output_3d = make_3d_for_bn(grad_output)
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    input_grad = torch.empty_like(input_3d)
    weight_grad = torch.empty_like(weight)
    bias_grad = torch.empty_like(weight)
    with torch_device_fn.device(input.device):
        count = batch_dim * spatial_dim
        if count <= 2048:
            _backward_fused[(feat_dim,)](
                input_3d,
                grad_output_3d,
                weight,
                save_mean,
                save_var,
                input_grad,
                weight_grad,
                bias_grad,
                count,
                feat_dim,
                spatial_dim,
                triton.next_power_of_2(count),
            )
        else:
            parts = triton.cdiv(count, 1024)
            partial = torch.empty(
                (feat_dim, 2, parts), dtype=input.dtype, device=input.device
            )
            _backward_partial[(feat_dim, parts)](
                input_3d,
                grad_output_3d,
                save_mean,
                partial,
                count,
                feat_dim,
                spatial_dim,
                parts,
                1024,
            )
            _backward_apply[(feat_dim, parts)](
                input_3d,
                grad_output_3d,
                weight,
                save_mean,
                save_var,
                partial,
                input_grad,
                weight_grad,
                bias_grad,
                count,
                feat_dim,
                spatial_dim,
                parts,
                triton.next_power_of_2(parts),
                1024,
            )
        return input_grad.view_as(input), weight_grad, bias_grad
