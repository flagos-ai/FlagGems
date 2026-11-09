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

"""GCU400 专属 margin_ranking_loss 实现。

由公共实现迁移而来：保留 autograd.Function 的广播、reduction 与反向语义，
逐元素 kernel 改为固定 grid + 核内跨步循环，减少 GCU400 上的 block 发射开销。
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# 固定 grid 上限取 GCU400 的 SIP 数量级，剩余 block 由核内跨步循环覆盖。
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
        # 差值必须保留输入 dtype 的舍入：aten 的 fp16/bf16 参考实现先按低精度
        # 相减，若在此处提升到 fp32 会让 max(0, ...) 边界与参考结果分歧。
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
        x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        loss = tl.maximum(margin - target * (x1 - x2), 0.0)
        tl.store(out_ptr + offsets, loss.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total", "margin"])
def _margin_ranking_loss_backward_kernel(
    grad_output_ptr,
    x1_ptr,
    x2_ptr,
    target_ptr,
    grad_x1_ptr,
    grad_x2_ptr,
    margin,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    element_offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(program_id, num_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + element_offsets
        mask = offsets < N_total
        # 与前向保持相同的低精度差值判定，避免 active_mask 在边界处错位。
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
    """margin ranking loss 的前向与反向，主计算路径由两个 Triton kernel 完成。"""

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
            # 同形同 dtype 的常见 case 直接展平，省去广播与 dtype 转换的 host 开销。
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

        if reduction == "mean":
            grad_output = grad_output.expand(n_elements) / n_elements
        elif reduction == "sum":
            grad_output = grad_output.expand(n_elements)
        else:
            grad_output = grad_output.contiguous().view(-1)
        grad_output = grad_output.contiguous()

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
                BLOCK_SIZE=BLOCK,
                num_warps=4,
            )

        original_shape = ctx.original_shape
        grad_x1 = grad_x1.view(original_shape)
        grad_x2 = grad_x2.view(original_shape)
        grad_target = torch.zeros_like(target).view(original_shape)
        return grad_x1, grad_x2, grad_target, None, None


def margin_ranking_loss(x1, x2, target, margin=0.0, reduction="mean"):
    """计算 loss = max(0, -target * (x1 - x2) + margin)，签名与公共实现一致。"""
    return MarginRankingLossOp.apply(x1, x2, target, margin, reduction)
