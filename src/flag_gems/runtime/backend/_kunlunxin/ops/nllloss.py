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

from flag_gems.ops.nll_loss_nd import (
    nll_loss_nd_backward_kernel,
    nll_loss_nd_forward_kernel,
)
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sum import sum as xpu_sum

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_n

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    tl.store(out_ptr + offsets_n, out, mask=mask_n)
    if reduction == 1:
        tl.store(ignore_wgt_tgt_ptr + offsets_n, wgt_tgt, mask=mask_n)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offsets_n
        out_grad = tl.load(out_grad_ptrs, mask=mask_n, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)
    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=ignore_mask)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    out_ptrs = out_ptr + offset_n * D + offset_d
    tl.store(out_ptrs, out, mask=mask_block)

    if reduction == 1:
        ignore_wgt_tgt_ptrs = ignore_wgt_tgt_ptr + offset_n * D + offset_d
        tl.store(ignore_wgt_tgt_ptrs, wgt_tgt, mask=mask_block)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offset_n * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=mask_block, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1
    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offset_n * C * D + tgt * D + offset_d
    tl.store(inp_grad_ptrs, inp_grad, mask=ignore_mask)


# Negative Log Likelihood Loss (NLLLoss)
#
# This loss function is used for training classification problems with C classes.
#
# Parameters:
# - input (Tensor):
#   - Expected to contain log-probabilities for each class.
#   - Shape can be either:
#     - (minibatch, C) for standard classification tasks.
#     - (minibatch, C, d1, d2, ..., dK) for K-dimensional inputs (e.g., per-pixel loss for 2D images).
#
# - target (Tensor):
#   - Should contain class indices in the range [0, C-1].
#   - If ignore_index is specified, this index can be outside the class range
#       and will be ignored in the loss computation.
#
# - weight (1D Tensor, optional):
#   - Assigns weight to each class, useful for unbalanced datasets.
#
# Reduction modes:
# - 'none': returns per-sample loss (shape: (N,)).
# - 'mean' (default): computes the mean of the weighted losses.
# - 'sum': computes the sum of the weighted losses.
#
# Mathematical description:
# - Unreduced loss:
#   l_n = -w_y_n * x_n, where w_c = weight[c] * 1{c != ignore_index}.
# - Reduced loss (depending on the specified reduction mode):
#   - mean: ℓ(x, y) = (1/N) * Σ(w_y_n * l_n)
#   - sum: ℓ(x, y) = Σ(l_n)


# 1d & 2d tensor
def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS_FWD")
    assert self.ndim <= 2, "Invalid input ndim"
    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    out = torch.empty(shape, dtype=self.dtype, device=self.device)
    ignore_weight_tgt = None
    if reduction == 1:
        ignore_weight_tgt = torch.empty(
            target.shape, dtype=self.dtype, device=self.device
        )

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_forward_kernel[grid](
            self,  # torch.Size([4096, 256])
            target,  # torch.Size([4096]), tensor([174, 125, 174,  ..., 216, 171, 120])
            weight,  # torch.Size([256])
            out,  # torch.Size([4096])
            ignore_weight_tgt,  # torch.Size([4096])
            ignore_index,  # 1
            N,  # 4096
            C,  # 256
            reduction,  # 0
            is_use_mask_zero=True,
        )

    # redution: 0-None, 1-mean, 2-sum
    if reduction == 0:
        output = out
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        total_out = xpu_sum(out)
        total_weight = xpu_sum(ignore_weight_tgt).to(self.dtype)
        output = (total_out / total_weight).to(self.dtype)
    else:
        total_out = xpu_sum(out)
        output = total_out.to(self.dtype)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS_BWD")
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.empty_like(self).contiguous()
    zero_(grad_input)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_backward_kernel[grid](
            grad_output,
            target,
            weight,
            grad_input,
            ignore_index,
            total_weight,
            N,
            C,
            reduction,
        )

    return grad_input


# 3d+ tensor
def nll_loss2d_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS2D_FWD")
    assert self.ndim >= 3, "Invalid input ndim"

    N, C = self.shape[0], self.shape[1]
    D = self.numel() // (N * C)
    assert target.numel() == N * D, "Invalid target size"

    target_orig_shape = target.shape
    self_flat = self.reshape(N, C, D).contiguous()
    target_flat = target.reshape(N, D).contiguous()
    weight = None if weight is None else weight.contiguous()

    out = torch.empty((N, D), dtype=self.dtype, device=self.device)
    ignore_weight_tgt = None
    if reduction == 1:
        ignore_weight_tgt = torch.empty((N, D), dtype=self.dtype, device=self.device)

    grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
    with torch_device_fn.device(self.device):
        nll_loss2d_forward_kernel[grid](
            self_flat,
            target_flat,
            weight,
            out,
            ignore_weight_tgt,
            ignore_index,
            N,
            C,
            D,
            reduction,
            is_use_mask_zero=True,
        )

    # redution: 0-None, 1-mean, 2-sum
    if reduction == 0:
        output = out.reshape(target_orig_shape)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        total_out = xpu_sum(out)
        total_weight = xpu_sum(ignore_weight_tgt).to(self.dtype)
        output = (total_out / total_weight).to(self.dtype)
    else:
        total_out = xpu_sum(out)
        output = total_out.to(self.dtype)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss2d_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS2D_BWD")
    N, C = self.shape[0], self.shape[1]
    D = self.numel() // (N * C)

    grad_output = grad_output.contiguous()
    target_flat = target.reshape(N, D).contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.empty_like(self).contiguous()
    zero_(grad_input)

    grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
    with torch_device_fn.device(self.device):
        nll_loss2d_backward_kernel[grid](
            grad_output,
            target_flat,
            weight,
            grad_input.reshape(N, C, D),
            ignore_index,
            total_weight,
            N,
            C,
            D,
            reduction,
        )

    return grad_input


def nll_loss_nd_forward(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
):
    logger.debug("GEMS_KUNLUNXIN NLL LOSS ND FWD")
    if input.dim() < 3:
        return nll_loss_forward(
            input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

    N = input.shape[0]
    C = input.shape[1]
    S = input.numel() // (N * C)
    inp = input.reshape(N, C, S)

    if target.numel() != N * S:
        raise ValueError(
            f"Target size {target.shape} doesn't match input size (N={N}, S={S})"
        )
    tgt = target.reshape(N, S)
    stride_in_n, stride_in_c, stride_in_s = inp.stride()
    stride_tgt_n, stride_tgt_s = tgt.stride()

    if weight is None:
        has_weight = False
        w = input
    else:
        has_weight = True
        if weight.numel() != C:
            raise ValueError(f"Weight shape {weight.shape} must be ({C},)")
        w = weight.contiguous()
    if reduction not in [0, 1, 2]:
        raise ValueError("reduction must be 0 ('none'), 1 ('mean'), or 2 ('sum')")

    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_S"]), N)
    with torch_device_fn.device(input.device):
        if reduction == 0:
            out = torch.empty((N, S), device=input.device, dtype=input.dtype)
            scratch = torch.empty(1, device=input.device)
            nll_loss_nd_forward_kernel[grid](
                inp, tgt, w, out, scratch, C, S,
                stride_in_n, stride_in_c, stride_in_s,
                stride_tgt_n, stride_tgt_s, ignore_index,
                HAS_WEIGHT=has_weight, REDUCTION=reduction,
            )
            res = out.view_as(target) if target.dim() == input.dim() - 1 else out.reshape(target.shape)
            return res, torch.empty([], device=input.device, dtype=input.dtype)

        out = torch.empty(1, device=input.device, dtype=input.dtype)
        scratch = torch.empty(3, device=input.device, dtype=torch.float32)
        zero_(scratch)
        nll_loss_nd_forward_kernel[grid](
            inp, tgt, w, out, scratch, C, S,
            stride_in_n, stride_in_c, stride_in_s,
            stride_tgt_n, stride_tgt_s, ignore_index,
            HAS_WEIGHT=has_weight, REDUCTION=reduction,
        )
        out = out[0]
        total_weight = scratch[1] if reduction == 1 else torch.empty([], device=input.device, dtype=input.dtype)
        return out, total_weight


def nll_loss_nd_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
    total_weight: torch.Tensor = None,
):
    logger.debug("GEMS_KUNLUNXIN NLL LOSS ND BWD")
    if input.dim() < 3:
        return nll_loss_backward(
            grad_output, input, target, weight=weight, reduction=reduction,
            ignore_index=ignore_index, total_weight=total_weight,
        )

    grad_input = torch.empty_like(input)
    zero_(grad_input)
    N = input.shape[0]
    C = input.shape[1]
    S = input.numel() // (N * C)
    grad_inp = grad_input.reshape(N, C, S)
    tgt = target.reshape(N, S)
    stride_in_n, stride_in_c, stride_in_s = grad_inp.stride()
    stride_tgt_n, stride_tgt_s = tgt.stride()
    if weight is None:
        has_weight = False
        w = input
    else:
        has_weight = True
        w = weight.contiguous()
    if reduction == 0:
        grad_out = grad_output.reshape(N, S)
        stride_go_n, stride_go_s = grad_out.stride()
    else:
        grad_out = grad_output
        stride_go_n, stride_go_s = 0, 0

    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_S"]), N)
    with torch_device_fn.device(input.device):
        nll_loss_nd_backward_kernel[grid](
            grad_out, tgt, w, grad_input, total_weight, C, S,
            stride_in_n, stride_in_c, stride_in_s,
            stride_tgt_n, stride_tgt_s, stride_go_n, stride_go_s,
            ignore_index, HAS_WEIGHT=has_weight, REDUCTION=reduction,
        )
    return grad_input
