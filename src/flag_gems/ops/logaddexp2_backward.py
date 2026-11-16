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


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, "DEFAULT"), (0, 2, "DEFAULT")],
)
@triton.jit
def logaddexp2_backward_func(grad_output, x, y):
    grad_f32 = grad_output.to(tl.float32)
    # ``other - self``, ``pow(2, .)`` and ``1 + .`` are separate ATen tensors in
    # the generated backward, so each is rounded to the element dtype.
    delta = (x.to(tl.float32) - y.to(tl.float32)).to(x.dtype).to(tl.float32)
    pow_self = tl.exp2(-delta).to(x.dtype).to(tl.float32)
    pow_other = tl.exp2(delta).to(y.dtype).to(tl.float32)
    denom_self = (1.0 + pow_self).to(x.dtype).to(tl.float32)
    denom_other = (1.0 + pow_other).to(y.dtype).to(tl.float32)
    return grad_f32 / denom_self, grad_f32 / denom_other


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_backward_self_func(grad_output, x, y):
    delta = (x.to(tl.float32) - y.to(tl.float32)).to(x.dtype).to(tl.float32)
    pow_delta = tl.exp2(-delta).to(x.dtype).to(tl.float32)
    denom = (1.0 + pow_delta).to(x.dtype).to(tl.float32)
    return grad_output.to(tl.float32) / denom


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 2, "DEFAULT")])
@triton.jit
def logaddexp2_backward_other_func(grad_output, x, y):
    delta = (x.to(tl.float32) - y.to(tl.float32)).to(y.dtype).to(tl.float32)
    pow_delta = tl.exp2(delta).to(y.dtype).to(tl.float32)
    denom = (1.0 + pow_delta).to(y.dtype).to(tl.float32)
    return grad_output.to(tl.float32) / denom


# Same-shape fast path. The generic ``pointwise_dynamic`` wrapper costs ~50 us
# of host time per call, which dominates a launch-bound backward on small
# tensors, so the common case (identical shapes, same dtype, contiguous)
# launches this dedicated kernel instead: one launch for the needed gradients,
# no per-call wrapper machinery.
@triton.jit
def _logaddexp2_backward_kernel(
    grad_ptr,
    self_ptr,
    other_ptr,
    grad_self_ptr,
    grad_other_ptr,
    n_elements,
    NEED_SELF: tl.constexpr,
    NEED_OTHER: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offsets = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    else:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # The generated ATen backward materializes ``other - self``, ``pow(2, .)``
    # and ``1 + .`` as separate tensors, so each is rounded to the element
    # dtype; repeating that order keeps this kernel bit-identical to native.
    if EVEN:
        x = tl.load(self_ptr + offsets)
        y = tl.load(other_ptr + offsets)
    else:
        mask = offsets < n_elements
        x = tl.load(self_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    if NEED_SELF:
        if EVEN:
            grad = tl.load(grad_ptr + offsets)
        else:
            grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        delta = (x.to(tl.float32) - y.to(tl.float32)).to(x.dtype).to(tl.float32)
        pow_delta = tl.exp2(-delta).to(x.dtype).to(tl.float32)
        denom = (1.0 + pow_delta).to(x.dtype).to(tl.float32)
        grad_input = grad.to(tl.float32) / denom
        if EVEN:
            tl.store(grad_self_ptr + offsets, grad_input)
        else:
            tl.store(grad_self_ptr + offsets, grad_input, mask=mask)

    if NEED_OTHER:
        if EVEN:
            grad = tl.load(grad_ptr + offsets)
        else:
            grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        delta = (x.to(tl.float32) - y.to(tl.float32)).to(y.dtype).to(tl.float32)
        pow_delta = tl.exp2(delta).to(y.dtype).to(tl.float32)
        denom = (1.0 + pow_delta).to(y.dtype).to(tl.float32)
        grad_input = grad.to(tl.float32) / denom
        if EVEN:
            tl.store(grad_other_ptr + offsets, grad_input)
        else:
            tl.store(grad_other_ptr + offsets, grad_input, mask=mask)


_BACKWARD_BLOCK = 1024
# Above this element count a CTA carries 8 warps: the very large shapes have
# enough resident CTAs that the extra warps hide DRAM latency, while smaller
# shapes prefer the shorter 4-warp CTA.
_BACKWARD_WIDE_WARP_ELEMENTS = 33554432
_BACKWARD_INT32_MAX = 2**31 - 1
_BACKWARD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _can_use_backward_fast_path(grad_output, self, other):
    return (
        self.shape == other.shape == grad_output.shape
        and self.dtype == other.dtype == grad_output.dtype
        and self.dtype in _BACKWARD_DTYPES
        and self.is_contiguous()
        and other.is_contiguous()
        and grad_output.is_contiguous()
    )


def _launch_backward_fast_path(grad_output, self, other, need_self, need_other):
    grad_self = torch.empty_like(self) if need_self else None
    grad_other = torch.empty_like(other) if need_other else None
    n_elements = self.numel()
    if n_elements == 0:
        return grad_self, grad_other
    grid = ((n_elements + _BACKWARD_BLOCK - 1) // _BACKWARD_BLOCK,)
    with torch_device_fn.device(self.device):
        _logaddexp2_backward_kernel[grid](
            grad_output,
            self,
            other,
            grad_self if need_self else grad_output,
            grad_other if need_other else grad_output,
            n_elements,
            NEED_SELF=need_self,
            NEED_OTHER=need_other,
            EVEN=n_elements % _BACKWARD_BLOCK == 0,
            I64=n_elements > _BACKWARD_INT32_MAX,
            BLOCK_SIZE=_BACKWARD_BLOCK,
            num_warps=8 if n_elements > _BACKWARD_WIDE_WARP_ELEMENTS else 4,
        )
    return grad_self, grad_other


def _unbroadcast(grad, shape):
    """Sum a broadcasted gradient back to ``shape`` with a Triton reduction."""
    if tuple(grad.shape) == tuple(shape):
        return grad
    # Imported lazily: sum_to_size pulls in the factored sum kernels, and this
    # module is imported early from flag_gems.ops.
    from flag_gems.ops.sum_to_size import sum_to_size

    return sum_to_size(grad, *shape)


def logaddexp2_backward(grad_output, self, other):
    """First-order gradients of ``logaddexp2(self, other)``.

    Mirrors the ATen autograd pair, including broadcasting of the inputs to the
    forward output shape followed by a reduction back to each input shape.
    """
    logger.debug("GEMS LOGADDEXP2_BACKWARD")
    if _can_use_backward_fast_path(grad_output, self, other):
        return _launch_backward_fast_path(grad_output, self, other, True, True)
    grad_self, grad_other = logaddexp2_backward_func(grad_output, self, other)
    return (
        _unbroadcast(grad_self, self.shape).to(self.dtype),
        _unbroadcast(grad_other, other.shape).to(other.dtype),
    )
