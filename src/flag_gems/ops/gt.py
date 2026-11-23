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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def gt_func(x, y):
    return x.to(tl.float32) > y


def gt(A, B):
    logger.debug("GEMS GT")
    return gt_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def gt_func_scalar(x, y):
    return x.to(tl.float32) > y


def gt_scalar(A, B):
    logger.debug("GEMS GT_SCALAR")
    return gt_func_scalar(A, B)


def _is_fp8(t):
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


class _GtInplace(torch.autograd.Function):
    """Record the in-place write and its zero comparison gradients."""

    @staticmethod
    def forward(ctx, receiver, other, scalar):
        ctx.mark_dirty(receiver)
        if receiver.numel():
            if scalar:
                gt_func_scalar(receiver, other, out0=receiver)
            else:
                gt_func(receiver, other, out0=receiver)
        if not scalar:
            ctx.other_shape = other.shape
            ctx.other_dtype = other.dtype
            ctx.other_device = other.device
        return receiver

    @staticmethod
    def backward(ctx, grad_output):
        grad_receiver = (
            torch.zeros_like(grad_output) if ctx.needs_input_grad[0] else None
        )
        grad_other = (
            torch.zeros(ctx.other_shape, device=ctx.other_device, dtype=ctx.other_dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        return grad_receiver, grad_other, None


def _gt_inplace(receiver, other, scalar):
    if torch.is_grad_enabled() and (
        receiver.requires_grad or (not scalar and other.requires_grad)
    ):
        # Reject leaves (and views of leaves) before writing, as native gt_ does.
        if receiver.requires_grad and receiver.is_leaf:
            raise RuntimeError(
                "a leaf Variable that requires grad is being used in an in-place operation."
            )
        if (
            receiver.requires_grad
            and receiver._base is not None
            and receiver._base.is_leaf
        ):
            raise RuntimeError(
                "a view of a leaf Variable that requires grad is being used in an in-place operation."
            )
        return _GtInplace.apply(receiver, other, scalar)

    if receiver.numel():
        if scalar:
            gt_func_scalar(receiver, other, out0=receiver)
        else:
            gt_func(receiver, other, out0=receiver)
    torch.autograd.graph.increment_version(receiver)
    return receiver


def gt_tensor_(A, B):
    logger.debug("GEMS GT__TENSOR")
    if _is_fp8(A):
        raise NotImplementedError("gt_ is not implemented for fp8 inputs")
    return _gt_inplace(A, B, False)


def gt_scalar_(A, B):
    logger.debug("GEMS GT__SCALAR")
    if _is_fp8(A):
        raise NotImplementedError("gt_ is not implemented for fp8 inputs")
    return _gt_inplace(A, B, True)
