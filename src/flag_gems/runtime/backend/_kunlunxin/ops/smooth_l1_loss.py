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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _l1_loss(input, target):
    return tl.abs(input.to(tl.float32) - target.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_loss(input, target, beta):
    diff = tl.abs(input.to(tl.float32) - target.to(tl.float32))
    return tl.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def _l1_backward(input, target, grad_output):
    diff = input.to(tl.float32) - target.to(tl.float32)
    grad = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    return grad * grad_output.to(tl.float32)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _l1_backward_scalar(input, target, grad_output):
    diff = input.to(tl.float32) - target.to(tl.float32)
    grad = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    return grad * grad_output


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_backward(input, target, grad_output, beta):
    diff = input.to(tl.float32) - target.to(tl.float32)
    sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    grad = tl.where(tl.abs(diff) < beta, diff / beta, sign)
    return grad * grad_output.to(tl.float32)


@pointwise_dynamic(
    is_tensor=[True, True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _smooth_backward_scalar(input, target, grad_output, beta):
    diff = input.to(tl.float32) - target.to(tl.float32)
    sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    grad = tl.where(tl.abs(diff) < beta, diff / beta, sign)
    return grad * grad_output


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        return {"none": 0, "mean": 1, "sum": 2}[reduction]
    return reduction


def _broadcast_inputs(input, target):
    shape = torch.broadcast_shapes(input.shape, target.shape)
    if input.numel() == 0 or target.numel() == 0:
        return shape, None, None
    return shape, input, target


def _loss_values(input, target, beta):
    if beta == 0.0:
        return _l1_loss(input, target)
    return _smooth_loss(input, target, beta)


def smooth_l1_loss(input, target, reduction=1, beta: float = 1.0):
    logger.debug("GEMS KUNLUNXIN SMOOTH_L1_LOSS")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)
    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    shape, input_expanded, target_expanded = _broadcast_inputs(input, target)
    if input_expanded is None:
        if reduction == 0:
            return torch.empty(shape, device=input.device, dtype=input.dtype)
        if reduction == 1:
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)
        return torch.zeros((), device=input.device, dtype=input.dtype)

    loss = _loss_values(input_expanded, target_expanded, beta)
    if reduction == 0:
        return loss
    result = torch.sum(loss)
    if reduction == 1:
        result = result / loss.numel()
    return result


def smooth_l1_loss_out(input, target, reduction=1, beta: float = 1.0, *, out):
    logger.debug("GEMS KUNLUNXIN SMOOTH_L1_LOSS OUT")
    result = smooth_l1_loss(input, target, reduction, beta)
    out.resize_(result.shape)
    out.copy_(result)
    return out


def smooth_l1_loss_backward(grad_output, input, target, reduction, beta: float):
    logger.debug("GEMS KUNLUNXIN SMOOTH_L1_LOSS BACKWARD")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)
    if beta < 0:
        raise RuntimeError("smooth_l1_loss does not support negative values for beta.")

    shape = torch.broadcast_shapes(input.shape, target.shape)
    if input.numel() == 0 or target.numel() == 0:
        return torch.empty(shape, device=input.device, dtype=input.dtype)

    if grad_output.numel() == 1:
        grad_scale = grad_output.item()
        if reduction == 1:
            grad_scale /= input.numel()
        if beta == 0.0:
            return _l1_backward_scalar(input, target, grad_scale)
        return _smooth_backward_scalar(input, target, grad_scale, beta)

    if beta == 0.0:
        return _l1_backward(input, target, grad_output)
    return _smooth_backward(input, target, grad_output, beta)
