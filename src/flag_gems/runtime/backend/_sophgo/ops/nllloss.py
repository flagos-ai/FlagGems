import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry



BLOCK_N = 256
BLOCK_ND = 128


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    sum_ptr,
    weight_ptr,
    ignore_index,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = BLOCK_N,
):

    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)


    mask_n = offsets_n < N


    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    assert tgt >= 0 and tgt < C, "Invalid target value"


    ignore_mask = not (tgt == ignore_index) and mask_n




    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:

        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)



    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)


    out = inp_tgt * wgt_tgt * -1



    if reduction == 0:
        tl.store(out_ptr + offsets_n, out, mask=mask_n)

    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        pid = tl.program_id(0)

        tl.store(sum_ptr + pid, total_out)
        tl.store(weight_ptr + pid, total_wgt)

    else:
        total_out = tl.sum(out)
        pid = tl.program_id(0)
        tl.store(sum_ptr + pid, total_out)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,               # batch size
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = BLOCK_N,
):

    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N


    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n


    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
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
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_n)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    sum_ptr,
    weight_ptr,
    ignore_index,
    N,              # batch size
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = BLOCK_ND,
):

    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)


    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D


    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    assert tgt >= 0 and tgt < C, "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_block


    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)


    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1


    if reduction == 0:
        out_ptrs = out_ptr + offset_n * D + offset_d
        tl.store(out_ptrs, out, mask=mask_block)
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        pid = tl.program_id(0)
        tl.store(sum_ptr + pid, total_out)
        tl.store(weight_ptr + pid, total_wgt)
    else:
        total_out = tl.sum(out)
        pid = tl.program_id(0)
        tl.store(sum_ptr + pid, total_out)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,               # batch size
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = BLOCK_ND,
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
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
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




def _host_reduce_mean(sum_buf, weight_buf, out_dtype, device):
    total_sum = float(sum_buf.cpu().sum())
    total_weight = float(weight_buf.cpu().sum())
    mean_val = float("nan") if total_weight == 0.0 else (total_sum / total_weight)
    output = torch.tensor(mean_val, dtype=out_dtype, device=device)
    total_weight_tensor = torch.tensor(total_weight, dtype=out_dtype, device=device)
    return output, total_weight_tensor


def _host_reduce_sum(sum_buf, out_dtype, device):
    total_sum = float(sum_buf.cpu().sum())
    return torch.tensor(total_sum, dtype=out_dtype, device=device)


# 1d & 2d tensor
def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    """
    NLL Loss1D2D

    :
        self: tensorshape(N, C)(C,)log-probabilities
        target: shape(N,)scalar
        weight: shape(C,)
        reduction: 0=noneper-sample loss1=mean2=sum
        ignore_index:

    :
        output: loss
        total_weight:
    """
    logging.debug("GEMS NLL Loss FWD")
    assert self.ndim <= 2, "Invalid input ndim"
    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"


    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    out_buf = None
    sum_buf = None
    weight_buf = None
    counter_buf = None
    mean_buf = None




    if reduction == 0:
        out_buf = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        num_programs = triton.cdiv(N, BLOCK_N)
        sum_buf = torch.empty(num_programs, dtype=torch.float32, device=self.device)
        weight_buf = torch.empty_like(sum_buf)
    else:
        num_programs = triton.cdiv(N, BLOCK_N)
        sum_buf = torch.empty(num_programs, dtype=torch.float32, device=self.device)


    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_forward_kernel[grid](
            self,
            target,
            weight,
            out_buf,
            sum_buf,
            weight_buf,
            ignore_index,
            N,
            C,
            reduction,
        )


    if reduction == 0:

        output = out_buf
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:

        output, total_weight = _host_reduce_mean(
            sum_buf, weight_buf, self.dtype, self.device
        )
    else:
        output = _host_reduce_sum(sum_buf, self.dtype, self.device)
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
    """
    NLL Loss

    :
        grad_output:
        self: forwardtensor
        target: forward
        weight: forward
        reduction: forwardreduction
        ignore_index: forwardignore_index
        total_weight: forwardmean

    :
        grad_input: shapeself
    """
    logging.debug("GEMS NLL Loss BWD")
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()


    grad_input = torch.zeros_like(self).contiguous()

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
    """
    NLL Loss

    :
        self: tensorshape(N, C, H, D)
        target: shape(N, 1, D)
        weight: shape(C,)
        reduction: 0=none1=mean2=sum
        ignore_index:

    :
        output: loss
        total_weight:
    """
    logging.debug("GEMS NLL Loss2d FWD")
    assert self.ndim == 4, "Invalid input ndim"

    shape = list(target.shape)
    N, C, _, D = self.shape
    assert shape == [N, 1, D], "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    out_buf = None
    sum_buf = None
    weight_buf = None



    if reduction == 0:
        out_buf = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        num_programs = triton.cdiv(N * D, BLOCK_ND)
        sum_buf = torch.empty(num_programs, dtype=torch.float32, device=self.device)
        weight_buf = torch.empty_like(sum_buf)
    else:
        num_programs = triton.cdiv(N * D, BLOCK_ND)
        sum_buf = torch.empty(num_programs, dtype=torch.float32, device=self.device)


    grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
    with torch_device_fn.device(self.device):
        nll_loss2d_forward_kernel[grid](
            self,
            target,
            weight,
            out_buf,
            sum_buf,
            weight_buf,
            ignore_index,
            N,
            C,
            D,
            reduction,
        )


    if reduction == 0:
        output = out_buf
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        output, total_weight = _host_reduce_mean(
            sum_buf, weight_buf, self.dtype, self.device
        )
    else:
        output = _host_reduce_sum(sum_buf, self.dtype, self.device)
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
    """
    NLL Loss 2D

    :
        grad_output:
        self: forwardtensorshape (N, C, H, D)
        target: forwardshape (N, 1, D)
        weight: forward
        reduction: forwardreduction
        ignore_index: forwardignore_index
        total_weight: forward

    :
        grad_input: shapeself
    """
    logging.debug("GEMS NLL Loss2d BWD")
    N, C, _, D = self.shape

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
    with torch_device_fn.device(self.device):
        nll_loss2d_backward_kernel[grid](
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

    return grad_input
