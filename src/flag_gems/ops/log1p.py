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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log1p_func(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


class Log1pFunction(torch.autograd.Function):
    """Autograd wrapper that gives log1p a fused first-order backward.

    The Triton backward does not construct a graph for higher-order gradients.

    ATen declares ``aten::log1p`` CompositeImplicitAutograd with a generated
    ``Log1PBackward0`` formula, so serving its CUDA implementation from
    FlagGems alone leaves the backward to ATen's composite ``1 + x`` add
    followed by a division.  Saving ``x`` and launching the fused Triton kernel
    keeps the whole autograd path inside FlagGems.  The registration pairs this
    wrapper with an Autograd fallthrough key, matching the ``clone`` and
    ``adaptive_avg_pool3d`` precedent.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return log1p_func(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return log1p_backward(grad_output, x)


def log1p(A):
    logger.debug("GEMS LOG1P")
    if torch.is_grad_enabled() and A.requires_grad:
        return Log1pFunction.apply(A)
    return log1p_func(A)


def log1p_out(A, out):
    logger.debug("GEMS LOG1P_OUT")
    return log1p_func(A, out0=out)


@triton.jit
def _log1p_backward_grad(grad_output, x):
    # ATen computes log1p_backward as `grad / (1 + self)` with two roundings:
    # `1 + x` is evaluated with float32 opmath and rounded back to the input
    # dtype first, then the quotient is evaluated with float32 opmath and
    # rounded to the output dtype. Reproducing that ordering keeps fp16, bf16
    # and fp32 results bitwise-equal to native autograd; dividing once in
    # float32 instead drifts by up to a few ulp on fp16/bf16.
    value = x if x.dtype == tl.float64 else x.to(tl.float32)
    denominator = (1.0 + value).to(x.dtype)
    grad = (
        grad_output if grad_output.dtype == tl.float64 else grad_output.to(tl.float32)
    )
    # Triton's `/` is a fast division; ATen uses IEEE round-to-nearest.
    return tl.math.div_rn(grad, denominator.to(grad.dtype)).to(grad_output.dtype)


_log1p_backward_strided = pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])(
    _log1p_backward_grad
)


@triton.jit
def _log1p_backward_kernel(
    grad_output, x, grad_input, n, BLOCK: tl.constexpr, LARGE: tl.constexpr
):
    pid = tl.program_id(0)
    if LARGE:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    grad = tl.load(grad_output + offsets, mask, other=0)
    value = tl.load(x + offsets, mask, other=1)
    tl.store(grad_input + offsets, _log1p_backward_grad(grad, value), mask)


def log1p_backward(grad_output, x):
    """Fused gradient of real log1p: ``grad_output / (1 + x)``."""
    if not (x.is_contiguous() and grad_output.is_contiguous()):
        # Strided inputs, zero-stride (expanded) gradients and broadcasting are
        # handled by the generated pointwise kernel.
        return _log1p_backward_strided(grad_output, x)
    grad_input = torch.empty_like(x)
    n = x.numel()
    if n:
        block_size = 1024
        with torch_device_fn.device(x.device):
            _log1p_backward_kernel[(triton.cdiv(n, block_size),)](
                grad_output,
                x,
                grad_input,
                n,
                BLOCK=block_size,
                LARGE=n > 2147483647,
            )
    return grad_input
