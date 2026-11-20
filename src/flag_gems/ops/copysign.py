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


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def _get_uint_dtype(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return tl.core.get_int_dtype(num_bits, False)


@tl.constexpr
def _get_sign_bit_mask(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def copysign_func(input, other):
    # Magnitude of input, sign of other
    abs_val = tl.abs(input)
    # Check sign bit of other (bitcast to unsigned int and check MSB)
    num_bits: tl.constexpr = input.dtype.primitive_bitwidth
    uint_dtype = _get_uint_dtype(num_bits)
    sign_bit_mask: tl.constexpr = _get_sign_bit_mask(num_bits)
    other_u = other.to(uint_dtype, bitcast=True)
    # Extract sign bit and check if it's set
    return tl.where((other_u & sign_bit_mask) != 0, -abs_val, abs_val)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def copysign_func_scalar(input, other):
    # The sign of a scalar `other` is known at trace time, so only the magnitude
    # is computed on device. Bitcasting the scalar itself is not possible: a
    # Python float is an fp32 constant while `input` may be fp16/bf16, and the
    # bit widths must match for `bitcast=True`.
    abs_val = tl.abs(input)
    return tl.where(other >= 0, abs_val, -abs_val)


def _copysign_common_dtype(input, other):
    # torch resolves copysign to the common type of its operands and, when that
    # is integral, promotes it to float32. Promoting here keeps the kernel's
    # sign-bit logic valid: an integral magnitude has no sign bit, so uint8
    # values >= 128 would otherwise be flipped, and abs(int8 min) would wrap.
    if isinstance(other, torch.Tensor):
        common = torch.promote_types(input.dtype, other.dtype)
    else:
        common = input.dtype
    if not common.is_floating_point:
        common = torch.float32
    return common


def _prepare_copysign_args(input, other):
    common = _copysign_common_dtype(input, other)
    input = input if input.dtype == common else input.to(common)
    if isinstance(other, torch.Tensor):
        other = other if other.dtype == common else other.to(common)
    return input, other


class CopysignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_for_backward(input)
        ctx.other = other
        input, other = _prepare_copysign_args(input, other)
        func = (
            copysign_func if isinstance(other, torch.Tensor) else copysign_func_scalar
        )
        return func(input, other)

    @staticmethod
    def backward(ctx, grad_output):
        # ``aten::copysign`` has no standalone backward operator, so the
        # gradient w.r.t. ``self`` runs a Triton kernel from
        # ``flag_gems.ops.copysign_backward``. Its VJP with respect to ``other``
        # is identically zero; ATen returns a zero tensor rather than None when
        # ``other`` requires grad, so mirror that here.
        (input,) = ctx.saved_tensors
        from flag_gems.ops.copysign_backward import copysign_backward

        grad_input = copysign_backward(grad_output, input, ctx.other)
        grad_other = None
        if isinstance(ctx.other, torch.Tensor) and ctx.other.requires_grad:
            grad_other = torch.zeros_like(
                ctx.other, memory_format=torch.contiguous_format
            )
        return grad_input, grad_other


def copysign(input, other, *, out=None):
    logger.debug("GEMS COPYSIGN")
    if torch.is_grad_enabled() and input.requires_grad:
        return CopysignFunction.apply(input, other)
    if isinstance(other, torch.Tensor):
        func = copysign_func
    else:
        func = copysign_func_scalar
    input, other = _prepare_copysign_args(input, other)
    return func(input, other)


def copysign_out(input, other, *, out=None):
    logger.debug("GEMS COPYSIGN_OUT")
    if isinstance(other, torch.Tensor):
        func = copysign_func
    else:
        func = copysign_func_scalar
    input, other = _prepare_copysign_args(input, other)
    if out is None:
        return func(input, other)
    func(input, other, out0=out)
    return out
