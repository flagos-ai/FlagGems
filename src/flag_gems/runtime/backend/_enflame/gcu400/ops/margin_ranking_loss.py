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

"""GCU400-specific margin_ranking_loss implementation.

Migrated from the common implementation: autograd.Function broadcast, reduction and backward semantics are kept,
the elementwise kernel uses a fixed grid + in-kernel stride loop to cut block launch overhead on GCU400;
backward's 0-dim scalar gradient for mean/sum reduction takes an in-kernel scaled broadcast fast path, avoiding a
host-side
stride-0 broadcast division tensor (enflame true_divide measured 22-30x slower on that shape).
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Fixed grid cap is set to the order of GCU400's SIP count; remaining blocks are covered by an in-kernel stride loop.
NUM_SIPS = 48
BLOCK = 8192


@libentry()
@triton.jit(do_not_specialize=["N_total", "margin"])
def _margin_ranking_loss_kernel(
    x1_ptr, x2_ptr, target_ptr, out_ptr, N_total, margin, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    element_offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(program_id, num_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + element_offsets
        mask = offsets < N_total
        # The difference must keep the input dtype's rounding: aten's fp16/bf16 reference subtracts in low precision
        # first, so promoting to fp32 here would make the max(0, ...) boundary diverge from the reference.
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
        x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        loss = tl.maximum(margin - target * (x1 - x2), 0.0)
        tl.store(out_ptr + offsets, loss.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total", "margin", "grad_scale"])
def _margin_ranking_loss_backward_kernel(
    grad_output_ptr,
    x1_ptr,
    x2_ptr,
    target_ptr,
    grad_x1_ptr,
    grad_x2_ptr,
    margin,
    N_total,
    grad_scale,
    BLOCK_SIZE: tl.constexpr,
    GRAD_IS_SCALAR: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    element_offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    if GRAD_IS_SCALAR:
        # Under mean/sum reduction the upstream gradient is a 0-dim scalar: do the 1/N scaling and broadcast inside the
        # kernel,
        # avoiding a host-side materialized stride-0 broadcast tensor (enflame's true_divide on
        # stride-0 inputs measured 22-30x slower than on dense tensors in the elementwise path).
        # Numeric path matches aten scalar division: divide in fp32 then round back to the gradient dtype.
        grad_scalar = tl.load(grad_output_ptr).to(tl.float32) / grad_scale
        grad_scalar = grad_scalar.to(grad_x1_ptr.dtype.element_ty)
    for block_id in tl.range(program_id, num_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + element_offsets
        mask = offsets < N_total
        # Keep the same low-precision difference decision as forward, avoiding active_mask mismatch at the boundary.
        if GRAD_IS_SCALAR:
            grad_output = grad_scalar
        else:
            grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
        x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        active_mask = margin - target * (x1 - x2) > 0
        grad_x1 = tl.where(active_mask, -target * grad_output, 0.0)
        grad_x2 = tl.where(active_mask, target * grad_output, 0.0)
        tl.store(
            grad_x1_ptr + offsets,
            grad_x1.to(grad_x1_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            grad_x2_ptr + offsets,
            grad_x2.to(grad_x2_ptr.dtype.element_ty),
            mask=mask,
        )


def _grid(n_elements):
    num_blocks = triton.cdiv(n_elements, BLOCK)
    if num_blocks <= 128:
        return num_blocks
    return NUM_SIPS


class MarginRankingLossOp(torch.autograd.Function):
    """Forward and backward of margin ranking loss; the main compute path is two Triton kernels."""

    @staticmethod
    def forward(ctx, x1, x2, target, margin=0.0, reduction="mean"):
        logger.debug("GEMS_ENFLAME MARGIN_RANKING_LOSS")

        if not (
            x1.is_floating_point()
            and x2.is_floating_point()
            and target.is_floating_point()
        ):
            raise ValueError("All inputs must be floating point tensors")

        if isinstance(reduction, int):
            reduction = {0: "none", 1: "mean", 2: "sum"}.get(reduction, "mean")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")

        device = x1.device
        if not (isinstance(device, torch.device) and device.type == flag_gems.device):
            return torch.ops.aten.margin_ranking_loss(
                x1,
                x2,
                target,
                float(margin),
                {"none": 0, "mean": 1, "sum": 2}[reduction],
            )

        if (
            x1.shape == x2.shape == target.shape
            and x1.dtype == x2.dtype == target.dtype
            and x1.is_contiguous()
            and x2.is_contiguous()
            and target.is_contiguous()
        ):
            # Common same-shape same-dtype cases are flattened directly, saving host broadcast and dtype conversion
            # overhead.
            x1_c, x2_c, tgt_c = x1.view(-1), x2.view(-1), target.view(-1)
            original_shape = x1.shape
        else:
            x1_b, x2_b, tgt_b = torch.broadcast_tensors(x1, x2, target)
            common_dtype = x1_b.dtype if x1_b.is_floating_point() else torch.float32
            x1_b = x1_b.to(dtype=common_dtype)
            x2_b = x2_b.to(dtype=common_dtype)
            tgt_b = tgt_b.to(dtype=common_dtype)
            x1_c = x1_b.contiguous().view(-1)
            x2_c = x2_b.contiguous().view(-1)
            tgt_c = tgt_b.contiguous().view(-1)
            original_shape = x1_b.shape

        out = torch.empty_like(x1_c)
        n_elements = out.numel()
        if n_elements == 0:
            if reduction == "none":
                return out.view(original_shape)
            elif reduction == "sum":
                return out.sum()
            return out.mean()

        with torch_device_fn.device(device):
            _margin_ranking_loss_kernel[(_grid(n_elements),)](
                x1_c,
                x2_c,
                tgt_c,
                out,
                n_elements,
                float(margin),
                BLOCK_SIZE=BLOCK,
                num_warps=4,
            )

        ctx.save_for_backward(x1_c, x2_c, tgt_c)
        ctx.reduction = reduction
        ctx.margin = margin
        ctx.n_elements = n_elements
        ctx.original_shape = original_shape

        if reduction == "none":
            return out.view(original_shape)
        elif reduction == "sum":
            return out.sum()
        return out.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_ENFLAME MARGIN_RANKING_LOSS_BACKWARD")

        x1, x2, target = ctx.saved_tensors
        margin = ctx.margin
        reduction = ctx.reduction
        n_elements = ctx.n_elements

        if n_elements == 0:
            grad_x1 = torch.zeros_like(x1)
            grad_x2 = torch.zeros_like(x2)
            grad_target = torch.zeros_like(target)
            return grad_x1, grad_x2, grad_target, None, None

        # Under mean/sum reduction the upstream gradient is a 0-dim scalar: pass the scalar pointer and scaling factor
        # directly to the kernel for in-kernel broadcast, avoiding host-side stride-0 broadcast division / full copies.
        if reduction == "mean":
            if grad_output.numel() == 1:
                grad_output = grad_output.contiguous()
                grad_is_scalar = True
                grad_scale = float(n_elements)
            else:
                grad_output = (grad_output.expand(n_elements) / n_elements).contiguous()
                grad_is_scalar = False
                grad_scale = 1.0
        elif reduction == "sum":
            if grad_output.numel() == 1:
                grad_output = grad_output.contiguous()
                grad_is_scalar = True
                grad_scale = 1.0
            else:
                grad_output = grad_output.expand(n_elements).contiguous()
                grad_is_scalar = False
                grad_scale = 1.0
        else:
            grad_output = grad_output.contiguous().view(-1)
            grad_is_scalar = False
            grad_scale = 1.0

        grad_x1 = torch.empty_like(x1)
        grad_x2 = torch.empty_like(x2)

        with torch_device_fn.device(x1.device):
            _margin_ranking_loss_backward_kernel[(_grid(n_elements),)](
                grad_output,
                x1,
                x2,
                target,
                grad_x1,
                grad_x2,
                float(margin),
                n_elements,
                grad_scale,
                BLOCK_SIZE=BLOCK,
                GRAD_IS_SCALAR=grad_is_scalar,
                num_warps=4,
            )

        original_shape = ctx.original_shape
        grad_x1 = grad_x1.view(original_shape)
        grad_x2 = grad_x2.view(original_shape)
        grad_target = torch.zeros_like(target).view(original_shape)
        return grad_x1, grad_x2, grad_target, None, None


def margin_ranking_loss(x1, x2, target, margin=0.0, reduction="mean"):
    """Compute loss = max(0, -target * (x1 - x2) + margin); signature matches the common implementation."""
    logger.debug("GEMS_ENFLAME MARGIN_RANKING_LOSS")
    return MarginRankingLossOp.apply(x1, x2, target, margin, reduction)
