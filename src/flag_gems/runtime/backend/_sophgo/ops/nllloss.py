import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# Kernel block size constants for 1D and 2D NLL Loss computation.
# Ensures host-side buffer allocation matches kernel launch grid.
BLOCK_N = 128  # Number of samples processed per program in 1D case
BLOCK_ND = 128  # Number of samples × spatial dimensions per program in 2D case


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
    inp_ptr,  # Input tensor pointer, shape: (N, C) or (C,), storing log-probabilities
    tgt_ptr,  # Target tensor pointer, shape: (N,) or scalar, storing class indices
    wgt_ptr,  # Weight tensor pointer, shape: (C,), optional, weight per class
    out_ptr,  # Output buffer pointer, only used when reduction=0, stores per-sample loss
    sum_ptr,  # Partial sum buffer pointer, each program writes its own sum result
    weight_ptr,  # Partial weight sum buffer pointer, each program writes its own weight result
    ignore_index,  # Target class index to ignore, this class does not participate in loss computation
    N,  # Batch size (number of samples)
    C,  # Number of classes
    reduction: tl.constexpr = 1,  # Reduction mode: 0=none, 1=mean, 2=sum
    BLOCK_N: tl.constexpr = BLOCK_N,  # Block size processed per program
):
    # Get current program ID and sample index range
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Boundary mask to prevent out-of-bounds access
    mask_n = offsets_n < N

    # Load target class indices
    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    assert tgt >= 0 and tgt < C, "Invalid target value"

    # Construct ignore mask: exclude ignore_index samples, and must be within valid range
    ignore_mask = not (tgt == ignore_index) and mask_n

    # Compute weight per sample
    # If no weight provided, use 1.0 for valid samples, 0.0 for ignored samples
    # Use tl.where to avoid unsupported bool→float type conversion
    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        # Load corresponding weight based on target class index
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    # Load log-probability from input tensor for the target class
    # inp_ptr layout: (N, C), use offsets_n * C + tgt for linear indexing
    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)

    # Compute NLL loss: -weight * input_log_prob
    out = inp_tgt * wgt_tgt * -1

    # Process output based on reduction mode
    # reduction=0: Keep per-sample loss, write directly to output buffer
    if reduction == 0:
        tl.store(out_ptr + offsets_n, out, mask=mask_n)
    # reduction=1 (mean): compute current block's partial sum and partial weight sum
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        pid = tl.program_id(0)
        # Each program writes results to an independent slot, avoiding unsupported atomic operations
        tl.store(sum_ptr + pid, total_out)
        tl.store(weight_ptr + pid, total_wgt)
    # reduction=2 (sum): only compute current block's partial sum
    else:
        total_out = tl.sum(out)
        pid = tl.program_id(0)
        tl.store(sum_ptr + pid, total_out)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,  # Output gradient pointer, from upstream backpropagation
    tgt_ptr,  # Target class index pointer
    wgt_ptr,  # Weight pointer (optional)
    inp_grad_ptr,  # Input gradient output pointer, same shape as forward input
    ignore_index,  # Ignored class index
    total_weight,  # Total weight used when reduction=mean
    N,  # Batch size
    C,  # Number of classes
    reduction: tl.constexpr = 1,  # Reduction mode
    BLOCK_N: tl.constexpr = BLOCK_N,
):
    # Get sample range processed by current program
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    # Load target classes and construct ignore mask
    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n

    # Load or construct weights
    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    # Load output gradient based on reduction mode
    if reduction == 0:
        # none mode: each sample has independent gradient
        out_grad_ptrs = out_grad_ptr + offsets_n
        out_grad = tl.load(out_grad_ptrs, mask=mask_n, other=0).to(tl.float32)
    else:
        # mean/sum mode: all samples share a single scalar gradient
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    # Mean mode needs to divide by total weight
    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    # Compute input gradient: -out_grad * weight / total_weight
    # For ignored samples, gradient is 0
    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)

    # Write gradient back to the corresponding target class position
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_n)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
    inp_ptr,  # Input tensor pointer, shape: (N, C, H, D)
    tgt_ptr,  # Target tensor pointer, shape: (N, 1, D)
    wgt_ptr,  # Weight pointer, shape: (C,)
    out_ptr,  # Output buffer, used when reduction=0
    sum_ptr,  # Partial sum buffer
    weight_ptr,  # Partial weight sum buffer
    ignore_index,  # Ignored class index
    N,  # Batch size
    C,  # Number of classes
    D,  # Spatial dimension size
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = BLOCK_ND,
):
    # In 2D case, each program handles a block in N*D space
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)

    # Split linear index into batch and spatial dimensions
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    # Load target: shape (N, 1, D) is flattened
    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    assert tgt >= 0 and tgt < C, "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_block

    # Load or construct weight
    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    # Load input: shape (N, C, H, D), index is n*C*D + tgt*D + d
    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    # Process output based on reduction mode
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
    out_grad_ptr,  # Output gradient pointer
    tgt_ptr,  # Target pointer
    wgt_ptr,  # Weight pointer
    inp_grad_ptr,  # Input gradient output pointer
    ignore_index,  # Ignored index
    total_weight,  # Total weight (used in mean mode)
    N,  # Batch size
    C,  # Number of classes
    D,  # Spatial dimension
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = BLOCK_ND,
):
    # Handle 2D backpropagation
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    # Load target and construct mask
    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_block

    # Load weight
    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    # Load output gradient
    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offset_n * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=mask_block, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    # Compute normalization factor
    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    # Compute and store input gradient
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
#   - mean: ℓ(x, y) = (1/N) * Σ(w_y_n * l_n)
#   - sum: ℓ(x, y) = Σ(l_n)


# 1d & 2d tensor
def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    """
    NLL Loss forward implementation (1D and 2D input).

    Args:
        self: Input tensor, shape (N, C) or (C,), containing log-probabilities.
        target: Target class indices, shape (N,) or scalar.
        weight: Optional class weights, shape (C,).
        reduction: 0=none (return per-sample loss), 1=mean (weighted average), 2=sum.
        ignore_index: Target class index to ignore.

    Returns:
        output: Computed loss value.
        total_weight: Total weight (used for backpropagation).
    """
    logging.debug("GEMS NLL Loss FWD")
    assert self.ndim <= 2, "Invalid input ndim"
    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"

    # Ensure input tensors are stored contiguously
    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    out_buf = None
    sum_buf = None
    weight_buf = None

    # Allocate buffers based on reduction mode
    # reduction=0: requires per-sample output buffer
    # reduction=1/2: requires per-program partial sum buffer
    if reduction == 0:
        out_buf = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        num_programs = triton.cdiv(N, BLOCK_N)
        sum_buf = torch.empty(num_programs, dtype=torch.float32, device=self.device)
        weight_buf = torch.empty_like(sum_buf)
    else:
        num_programs = triton.cdiv(N, BLOCK_N)
        sum_buf = torch.empty(num_programs, dtype=torch.float32, device=self.device)

    # Launch kernel, grid size calculated from number of samples and block size
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

    # Process kernel output, complete final reduction
    if reduction == 0:
        # none mode: return per-sample loss directly
        output = out_buf
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        # mean mode: accumulate all program partial sums on host, compute weighted average
        total_sum = sum_buf.sum()
        total_weight = weight_buf.sum()
        mean_val = total_sum / total_weight
        output = mean_val.to(self.dtype)
        total_weight = total_weight.to(self.dtype)
    else:
        # sum mode: accumulate all program partial sums on host
        output = sum_buf.sum().to(self.dtype)
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
    NLL Loss backward implementation.

    Args:
        grad_output: Gradient from upstream.
        self: Input tensor from forward pass.
        target: Target classes from forward pass.
        weight: Weights from forward pass.
        reduction: Reduction mode used in forward pass.
        ignore_index: Ignore index used in forward pass.
        total_weight: Total weight returned from forward pass (needed for mean mode).

    Returns:
        grad_input: Input gradient, same shape as self.
    """
    logging.debug("GEMS NLL Loss BWD")
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    # Initialize gradient buffer to 0
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
    NLL Loss forward implementation for high-dimensional input (e.g., image segmentation tasks).

    Args:
        self: Input tensor, shape (N, C, H, D).
        target: Target class indices, shape (N, 1, D).
        weight: Optional class weights, shape (C,).
        reduction: 0=none, 1=mean, 2=sum.
        ignore_index: Target class index to ignore.

    Returns:
        output: Computed loss value.
        total_weight: Total weight (used for backpropagation).
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

    # Allocate buffers based on reduction mode
    # For 2D case, total elements is N*D
    if reduction == 0:
        out_buf = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        num_programs = triton.cdiv(N * D, BLOCK_ND)
        sum_buf = torch.empty(num_programs, dtype=torch.float32, device=self.device)
        weight_buf = torch.empty_like(sum_buf)
    else:
        num_programs = triton.cdiv(N * D, BLOCK_ND)
        sum_buf = torch.empty(num_programs, dtype=torch.float32, device=self.device)

    # Launch kernel, grid calculated based on N*D and block size
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

    # Process kernel output, complete final reduction
    if reduction == 0:
        output = out_buf
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        total_sum = sum_buf.sum()
        total_weight = weight_buf.sum()
        mean_val = total_sum / total_weight
        output = mean_val.to(self.dtype)
        total_weight = total_weight.to(self.dtype)
    else:
        output = sum_buf.sum().to(self.dtype)
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
    NLL Loss 2D backward implementation.

    Args:
        grad_output: Gradient from upstream.
        self: Input tensor from forward pass, shape (N, C, H, D).
        target: Target classes from forward pass, shape (N, 1, D).
        weight: Weights from forward pass.
        reduction: Reduction mode used in forward pass.
        ignore_index: Ignore index used in forward pass.
        total_weight: Total weight returned from forward pass.

    Returns:
        grad_input: Input gradient, same shape as self.
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
