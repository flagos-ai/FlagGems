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

from flag_gems.runtime import device
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


@triton.jit
def _maximum_values(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)
    return tl.maximum(X, Y, propagate_nan=tl.PropagateNan.ALL)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def maximum_kernel(X, Y):
    return _maximum_values(X, Y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def maximum_kernel_tensor_scalar(X, Y):
    return _maximum_values(X, Y.to(X.dtype))


@triton.jit
def _maximum_grad_values(X, Y, dOut, IS_SELF: tl.constexpr):
    """ATen's element-wise maximum gradient, evaluated in one pass.

    ATen derives both operands from ``where(self == other, grad / 2, grad)`` and
    then zeroes the strictly smaller side: ties split the upstream gradient,
    equal values therefore receive ``grad / 2`` from both directions, and NaN
    operands keep the whole gradient on both sides because ``==``, ``<`` and
    ``>`` are all false for them.  ``dOut * 0.5`` is bit-identical to ATen's
    ``grad / 2`` because half of a binary float is exact.
    """
    grad = tl.where(X == Y, dOut * 0.5, dOut)
    if IS_SELF:
        loser = X < Y
    else:
        loser = X > Y
    return tl.where(loser, tl.zeros_like(grad), grad)


# The two gradients share the comparisons above, so the common case where both
# operands require grad is a single kernel with two outputs.  Pointwise
# broadcasting is resolved by the generated wrapper, and the caller reduces
# broadcast operands afterwards.
@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2
)
@triton.jit
def maximum_backward_kernel(X, Y, dOut):
    return _maximum_grad_values(X, Y, dOut, True), _maximum_grad_values(
        X, Y, dOut, False
    )


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def maximum_backward_self_kernel(X, Y, dOut):
    return _maximum_grad_values(X, Y, dOut, True)


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def maximum_backward_other_kernel(X, Y, dOut):
    return _maximum_grad_values(X, Y, dOut, False)


# One fixed launch configuration for the dense fast path.  A sweep over
# BLOCK in {512, 1024, 2048, 4096} x num_warps in {2, 4, 8} put every
# configuration within 0-1.9% of the per-shape best (median 0.6%), i.e. inside
# the run-to-run noise, because the kernel is a pure 3-read/2-write stream at
# the HBM ceiling.  Autotuning would therefore buy nothing measurable while
# costing a transient pair of output-sized buffers and extra host work per call.
_MAXIMUM_BACKWARD_BLOCK = 1024
_MAXIMUM_BACKWARD_WARPS = 4


@triton.jit
def _maximum_backward_flat_kernel(
    X,
    Y,
    dOut,
    gX,
    gY,
    n_elements,
    SELF_GRAD: tl.constexpr,
    OTHER_GRAD: tl.constexpr,
    BLOCK: tl.constexpr,
    LARGE: tl.constexpr,
):
    """One flat pass over already-contiguous operands of identical shape.

    ``SELF_GRAD``/``OTHER_GRAD`` are constexpr so a single-sided graph compiles a
    kernel that never stores through the unused output pointer.
    """
    pid = tl.program_id(0)
    if LARGE:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    grad = tl.load(dOut + offsets, mask=mask)
    base = tl.where(x == y, grad * 0.5, grad)
    zero = tl.zeros_like(base)
    if SELF_GRAD:
        tl.store(gX + offsets, tl.where(x < y, zero, base), mask=mask)
    if OTHER_GRAD:
        tl.store(gY + offsets, tl.where(x > y, zero, base), mask=mask)


def _flat_maximum_grad(grad_output, X, Y, needs_X, needs_Y):
    grad_X = (
        torch.empty_like(X, memory_format=torch.contiguous_format) if needs_X else None
    )
    grad_Y = (
        torch.empty_like(Y, memory_format=torch.contiguous_format) if needs_Y else None
    )
    n_elements = X.numel()
    if n_elements:
        grid = ((n_elements + _MAXIMUM_BACKWARD_BLOCK - 1) // _MAXIMUM_BACKWARD_BLOCK,)
        # An unused output slot reuses an input pointer; the constexpr guard keeps
        # the kernel from ever storing through it.
        _maximum_backward_flat_kernel[grid](
            X,
            Y,
            grad_output,
            grad_X if needs_X else X,
            grad_Y if needs_Y else Y,
            n_elements,
            SELF_GRAD=needs_X,
            OTHER_GRAD=needs_Y,
            BLOCK=_MAXIMUM_BACKWARD_BLOCK,
            LARGE=n_elements > 2147483647,
            num_warps=_MAXIMUM_BACKWARD_WARPS,
        )
    return grad_X, grad_Y


def _flat_maximum_grad_applicable(grad_output, X, Y):
    return (
        X.shape == Y.shape == grad_output.shape
        and X.dtype == Y.dtype == grad_output.dtype
        and X.is_contiguous()
        and Y.is_contiguous()
        and grad_output.is_contiguous()
    )


def _match_operand(grad, operand):
    """Reduce a broadcast gradient and restore the operand's dtype."""
    if grad is None:
        return None
    if grad.shape != operand.shape:
        if grad.numel() == 0:
            return torch.zeros_like(operand)
        from flag_gems.ops.sum_to_size import sum_to_size

        grad = sum_to_size(grad, *operand.shape)
    if grad.dtype != operand.dtype:
        grad = grad.to(operand.dtype)
    return grad


class MaximumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return maximum_kernel(X, Y)

    @staticmethod
    def backward(ctx, grad_output):
        X, Y = ctx.saved_tensors
        needs_X, needs_Y = ctx.needs_input_grad
        if not needs_X and not needs_Y:
            return None, None
        grad_X = grad_Y = None
        if _flat_maximum_grad_applicable(grad_output, X, Y):
            grad_X, grad_Y = _flat_maximum_grad(grad_output, X, Y, needs_X, needs_Y)
        elif needs_X and needs_Y:
            grad_X, grad_Y = maximum_backward_kernel(X, Y, grad_output)
        elif needs_X:
            grad_X = maximum_backward_self_kernel(X, Y, grad_output)
        else:
            grad_Y = maximum_backward_other_kernel(X, Y, grad_output)
        return _match_operand(grad_X, X), _match_operand(grad_Y, Y)


def _check_backend(tensor):
    if tensor.device.type != device:
        raise RuntimeError(
            f"maximum(): expected a tensor on the {device} backend, but got "
            f"{tensor.device}"
        )


def _is_cpu_0d_tensor(tensor):
    return tensor.dim() == 0 and tensor.device.type == "cpu"


def _host_scalar(scalar, result_dtype):
    # Keep integral scalars as Python ints so int64/int32 values reach the
    # kernel without a detour through fp32.
    value = scalar.item()
    if result_dtype is torch.bool:
        return bool(value)
    if not scalar.dtype.is_floating_point:
        return int(value)
    return float(value)


def _maximum_with_cpu_scalar(scalar, tensor):
    _check_backend(tensor)
    # ATen treats a 0-dim operand as a wrapped scalar, so promotion must use
    # the original tensor, not the Python scalar.
    result_dtype = torch.result_type(scalar, tensor)
    # The scalar kernel promotes against a plain Python value, which no longer
    # carries the wrapped scalar's dtype; align the tensor with the already
    # resolved result dtype first.
    if tensor.dtype != result_dtype:
        tensor = tensor.to(result_dtype)
    # Python floats reach Triton as fp32. Materialise an fp64 scalar on the
    # device so its low mantissa bits survive.
    if scalar.dtype == torch.float64:
        return maximum_kernel(tensor, scalar.to(tensor))
    return maximum_kernel_tensor_scalar(tensor, _host_scalar(scalar, result_dtype))


def maximum(X, Y):
    logger.debug("GEMS MAXIMUM")
    if X.device == Y.device:
        _check_backend(X)
        # The backward kernels are only defined for floating-point operands of
        # the same device; anything else (mixed int/float, wrapped CPU scalars)
        # keeps the forward-only path it had before.
        if (
            torch.is_grad_enabled()
            and (X.requires_grad or Y.requires_grad)
            and X.is_floating_point()
            and Y.is_floating_point()
        ):
            return MaximumFunction.apply(X, Y)
        return maximum_kernel(X, Y)
    if _is_cpu_0d_tensor(X):
        return _maximum_with_cpu_scalar(X, Y)
    if _is_cpu_0d_tensor(Y):
        return _maximum_with_cpu_scalar(Y, X)
    raise RuntimeError(
        "Expected all tensors to be on the same device, but found at least "
        f"two devices, {X.device} and {Y.device}!"
    )
