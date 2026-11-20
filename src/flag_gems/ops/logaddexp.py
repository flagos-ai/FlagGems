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
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# Elements per program for the flat backward kernel. The backward is purely
# bandwidth bound (three reads, two writes per element), so the launch keeps a
# 4-element vector per lane with the default four warps. This tile was selected
# by a bounded sweep over {(512, 1024, 2048, 4096) x (4, 8) warps} on H800.
_BACKWARD_BLOCK_SIZE = 1024
_BACKWARD_NUM_WARPS = 4


@triton.jit
def _logaddexp_grads(grad, x, y):
    """ATen's ``logaddexp`` gradient formula in float32 opmath.

    ``d/dx = grad / (1 + exp(y - x))`` and ``d/dy = grad / (1 + exp(x - y))``,
    as recorded in PyTorch's ``derivatives.yaml``. Evaluating it in float32 and
    letting ``exp()`` saturate to inf/0 exactly like the reference keeps
    extreme differences and same-sign infinities bit-compatible with native.
    """
    delta = x - y
    return grad / (1.0 + tl.exp(-delta)), grad / (1.0 + tl.exp(delta))


@triton.jit
def _logaddexp_backward_flat_kernel(
    grad_ptr,
    x_ptr,
    y_ptr,
    grad_x_ptr,
    grad_y_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Mask-free flat backward for element counts that are a whole tile.

    Both input gradients are produced in one pass, so every element is read
    three times and written twice -- the minimum traffic for this operator, and
    half the reads of a per-operand formulation. Dropping the bounds mask from
    the full tiles removes the predicate arithmetic from the inner loop.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    grad = tl.load(grad_ptr + offsets).to(tl.float32)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    y = tl.load(y_ptr + offsets).to(tl.float32)

    grad_x, grad_y = _logaddexp_grads(grad, x, y)

    tl.store(
        grad_x_ptr + offsets,
        grad_x.to(grad_x_ptr.dtype.element_ty),
    )
    tl.store(
        grad_y_ptr + offsets,
        grad_y.to(grad_y_ptr.dtype.element_ty),
    )


@triton.jit
def _logaddexp_backward_tail_kernel(
    grad_ptr,
    x_ptr,
    y_ptr,
    grad_x_ptr,
    grad_y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Masked tail for element counts that are not a whole number of tiles."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    grad_x, grad_y = _logaddexp_grads(grad, x, y)

    tl.store(
        grad_x_ptr + offsets,
        grad_x.to(grad_x_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        grad_y_ptr + offsets,
        grad_y.to(grad_y_ptr.dtype.element_ty),
        mask=mask,
    )


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(x, y):
    # log(exp(x) + exp(y)) = m + log(1 + exp(-|x - y|)), m = max(x, y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    return m + tl.log(1.0 + tl.exp(-tl.abs(delta)))


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
)
@triton.jit
def logaddexp_backward_func(grad_output, x, y):
    # Broadcast-, stride- and dtype-aware variant used by the autograd wrapper
    # whenever the dense flat kernel does not apply.
    return _logaddexp_grads(
        grad_output.to(tl.float32), x.to(tl.float32), y.to(tl.float32)
    )


def _reduce_to_shape(grad, shape):
    """Sum a broadcasted gradient back to the shape of its input."""
    if tuple(grad.shape) == tuple(shape):
        return grad
    # Imported lazily to keep the op module import graph small and avoid a
    # hard dependency on the reduction kernel for the common same-shape path.
    from flag_gems.ops.sum_to_size import sum_to_size

    return sum_to_size(grad, tuple(shape))


def _flat_backward_supported(grad_output, x, y):
    """Whether the fused flat kernel covers this call.

    The flat kernel addresses every operand linearly, so it needs three
    identically shaped, identically typed, dense operands. Anything else
    (broadcasting, mixed dtypes, non-contiguous views) uses the generic
    pointwise kernel, which resolves the broadcast and strides for us.
    """
    return (
        x.shape == y.shape == grad_output.shape
        and x.dtype == y.dtype == grad_output.dtype
        and x.device == y.device == grad_output.device
        and x.dtype.is_floating_point
        and x.is_contiguous()
        and y.is_contiguous()
        and grad_output.is_contiguous()
        and x.numel() <= 2**31 - 1
    )


def _launch_flat_backward(grad_output, x, y):
    grad_x = torch.empty_like(x)
    grad_y = torch.empty_like(y)
    n_elements = x.numel()
    if n_elements == 0:
        return grad_x, grad_y

    with torch_device_fn.device(x.device):
        if n_elements % _BACKWARD_BLOCK_SIZE == 0:
            _logaddexp_backward_flat_kernel[(n_elements // _BACKWARD_BLOCK_SIZE,)](
                grad_output,
                x,
                y,
                grad_x,
                grad_y,
                BLOCK_SIZE=_BACKWARD_BLOCK_SIZE,
                num_warps=_BACKWARD_NUM_WARPS,
            )
        else:
            _logaddexp_backward_tail_kernel[
                (triton.cdiv(n_elements, _BACKWARD_BLOCK_SIZE),)
            ](
                grad_output,
                x,
                y,
                grad_x,
                grad_y,
                n_elements,
                BLOCK_SIZE=_BACKWARD_BLOCK_SIZE,
                num_warps=_BACKWARD_NUM_WARPS,
            )
    return grad_x, grad_y


def _logaddexp_backward_values(grad_output, x, y):
    """Return both gradients in the broadcast output shape."""
    logger.debug("GEMS LOGADDEXP_BACKWARD")
    if _flat_backward_supported(grad_output, x, y):
        return _launch_flat_backward(grad_output, x, y)
    return logaddexp_backward_func(grad_output, x, y)


def logaddexp_backward(grad_output, x, y):
    """Return first-order real gradients reduced to the original operand shapes."""
    if grad_output.is_complex() or x.is_complex() or y.is_complex():
        raise NotImplementedError(
            "FlagGems logaddexp backward does not support complex tensors"
        )
    grad_x, grad_y = _logaddexp_backward_values(grad_output, x, y)
    return _reduce_to_shape(grad_x, x.shape).to(x.dtype), _reduce_to_shape(
        grad_y, y.shape
    ).to(y.dtype)


class LogaddexpFunction(torch.autograd.Function):
    """Autograd wrapper for the differentiable ``aten::logaddexp`` op.

    The forward keeps the existing pointwise kernel; the backward reuses the
    saved operands and launches one fused Triton kernel that produces both
    input gradients. Broadcasted inputs are reduced back to their original
    shapes with the FlagGems ``sum_to_size`` kernel.
    """

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return logaddexp_func(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        need_x, need_y = ctx.needs_input_grad
        if not (need_x or need_y):
            return None, None

        grad_x_b, grad_y_b = _logaddexp_backward_values(grad_output, x, y)

        grad_x = None
        if need_x:
            grad_x = _reduce_to_shape(grad_x_b, x.shape)
            if grad_x.dtype != x.dtype:
                grad_x = grad_x.to(x.dtype)

        grad_y = None
        if need_y:
            grad_y = _reduce_to_shape(grad_y_b, y.shape)
            if grad_y.dtype != y.dtype:
                grad_y = grad_y.to(y.dtype)

        return grad_x, grad_y


def logaddexp(self, other):
    logger.debug("GEMS LOGADDEXP")
    if self.is_complex() or other.is_complex():
        raise NotImplementedError("FlagGems logaddexp does not support complex inputs")
    # Keep the plain pointwise kernel on the inference path so no autograd
    # bookkeeping is paid when no input can receive a gradient. This mirrors
    # ATen, which only builds a graph for differentiable inputs.
    if torch.is_grad_enabled() and (
        (self.is_floating_point() and self.requires_grad)
        or (other.is_floating_point() and other.requires_grad)
    ):
        return LogaddexpFunction.apply(self, other)
    return logaddexp_func(self, other)


def logaddexp_out(self, other, out):
    logger.debug("GEMS LOGADDEXP_OUT")
    logaddexp_func(self, other, out0=out)
    return out
