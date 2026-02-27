import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_nd_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    total_weight_ptr,
    ignore_index,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    """
    N-dimensional NLL Loss forward kernel.

    Input shape: (N, C, d1, d2, ..., dk) where D = d1 * d2 * ... * dk
    Target shape: (N, d1, d2, ..., dk) where D = d1 * d2 * ... * dk

    Memory layout:
    - inp: inp[n, c, d] = inp_ptr[n * C * D + c * D + d]
    - tgt: tgt[n, d] = tgt_ptr[n * D + d]
    """
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

    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    # none
    if reduction == 0:
        out_ptrs = out_ptr + offset_n * D + offset_d
        tl.store(out_ptrs, out, mask=mask_block)
    # mean
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        tl.atomic_add(out_ptr, total_out, sem="relaxed")  # output
        tl.atomic_add(total_weight_ptr, total_wgt, sem="relaxed")  # weight
        tl.atomic_add(total_weight_ptr + 1, 1, sem="release")  # counter
        counter = tl.load(total_weight_ptr + 1)
        if counter == tl.num_programs(0):
            total_out = tl.load(out_ptr)
            total_wgt = tl.load(total_weight_ptr)
            tl.store(out_ptr, total_out / total_wgt)
    # sum
    else:
        total_out = tl.sum(out)
        tl.atomic_add(out_ptr, total_out, sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_nd_backward_kernel(
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
    """
    N-dimensional NLL Loss backward kernel.
    """
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
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_block)


class NLLLossND(torch.autograd.Function):
    """
    Autograd Function for N-dimensional NLL Loss.
    """

    @staticmethod
    def forward(ctx, self, target, weight, reduction, ignore_index):
        logger.debug("GEMS NLL Loss ND")

        # Handle input dimensions
        # Input: (N, C) or (N, C, d1, d2, ..., dk)
        # Target: (N,) or (N, d1, d2, ..., dk)
        ndim = self.ndim
        assert ndim >= 2, "Input must have at least 2 dimensions"

        N = self.shape[0]
        C = self.shape[1]

        if ndim == 2:
            # (N, C) -> D = 1
            D = 1
        else:
            # (N, C, d1, d2, ..., dk) -> D = d1 * d2 * ... * dk
            D = 1
            for i in range(2, ndim):
                D *= self.shape[i]

        target_shape = list(target.shape)

        # Validate target shape
        expected_target_numel = N * D
        assert (
            target.numel() == expected_target_numel
        ), f"Invalid target size: got {target.numel()}, expected {expected_target_numel}"

        inp = self.contiguous()
        target = target.contiguous()
        weight = None if weight is None else weight.contiguous()

        # Output shape
        if reduction == 0:
            out = torch.empty(target_shape, dtype=inp.dtype, device=inp.device)
            total_weight = torch.empty([], dtype=inp.dtype, device=inp.device)
        elif reduction == 1:
            out = torch.zeros([], dtype=torch.float32, device=inp.device)
            total_weight = torch.zeros([2], dtype=torch.float32, device=inp.device)
        else:
            out = torch.zeros([], dtype=torch.float32, device=inp.device)
            total_weight = torch.empty([], dtype=inp.dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
        with torch_device_fn.device(inp.device):
            nll_loss_nd_kernel[grid](
                inp, target, weight, out, total_weight, ignore_index, N, C, D, reduction
            )

        # Process output based on reduction
        if reduction == 0:
            output = out
        elif reduction == 1:
            output = out.to(inp.dtype)
            total_weight = total_weight[0:1].view([])  # Extract just the weight
        else:
            output = out.to(inp.dtype)

        # Save for backward
        ctx.save_for_backward(inp, target, weight, total_weight)
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.N = N
        ctx.C = C
        ctx.D = D

        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS NLL Loss ND BWD")

        inp, target, weight, total_weight = ctx.saved_tensors
        reduction = ctx.reduction
        ignore_index = ctx.ignore_index
        N = ctx.N
        C = ctx.C
        D = ctx.D

        grad_output = grad_output.contiguous()

        grad_input = torch.zeros_like(inp).contiguous()

        grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
        with torch_device_fn.device(inp.device):
            nll_loss_nd_backward_kernel[grid](
                grad_output,
                target,
                weight,
                grad_input,
                ignore_index,
                total_weight,
                N,
                C,
                D,
                reduction,
            )

        return grad_input, None, None, None, None


def nll_loss_nd(self, target, weight=None, reduction=1, ignore_index=-100):
    """
    N-dimensional NLL Loss.

    Computes the negative log likelihood loss for N-dimensional inputs.

    Args:
        self: Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk)
              Expected to contain log-probabilities.
        target: Target tensor of shape (N,) or (N, d1, d2, ..., dk)
                Contains class indices in range [0, C-1].
        weight: Optional weight tensor of shape (C,) for class weighting.
        reduction: Reduction mode (0=none, 1=mean, 2=sum).
        ignore_index: Target value to ignore in loss computation.

    Returns:
        Loss tensor.
    """
    return NLLLossND.apply(self, target, weight, reduction, ignore_index)
