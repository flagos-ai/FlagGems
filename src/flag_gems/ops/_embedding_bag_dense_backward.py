import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _embedding_bag_dense_backward_kernel(
    grad_ptr,
    indices_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    mode,
    has_per_sample_weights,
    per_sample_weights_ptr,
    scale_grad_by_freq,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    """Kernel for _embedding_bag_dense_backward.

    Args:
        grad_ptr: gradient tensor of shape (num_bags, embedding_dim)
        indices_ptr: indices into embedding table, shape (num_samples,)
        offset2bag_ptr: maps sample position to bag index, shape (num_samples,)
        bag_size_ptr: number of samples in each bag, shape (num_bags,)
        grad_weight_ptr: output gradient for embedding table, shape (num_weights, embedding_dim)
        num_weights: number of rows in embedding table
        padding_idx: padding index to ignore
        mode: 0=mean, 1=sum
        has_per_sample_weights: whether per_sample_weights are provided
        per_sample_weights_ptr: optional per-sample weights
        scale_grad_by_freq: whether to scale by word frequency
        BLOCK_D: block size for embedding dimension
        EMBED_DIM: embedding dimension

    Note on scaling:
        For mode=0 (mean): grad is already scaled by bag_size, so no division needed
        For mode=1 (sum): grad needs to be divided by bag_size
    """
    pid = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    # Load index and bag for this sample
    idx = tl.load(indices_ptr + pid).to(tl.int32)
    bag_idx = tl.load(offset2bag_ptr + pid).to(tl.int32)

    # Check validity
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    # Load bag size for this bag
    bag_size = tl.load(bag_size_ptr + bag_idx).to(tl.float32)
    bag_size = tl.where(bag_size == 0.0, 1.0, bag_size)

    # Load gradient for this bag
    go_ptrs = grad_ptr + bag_idx * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    # Compute scale factor based on mode
    # For mode=0 (mean): grad is already scaled by bag_size from autograd, so scale=1
    # For mode=1 (sum): grad needs to be divided by bag_size
    if mode == 0:
        scale = 1.0
    else:
        # Sum mode: divide by bag size
        scale = 1.0 / bag_size

    # Handle per_sample_weights if provided
    if has_per_sample_weights:
        pw = tl.load(per_sample_weights_ptr + pid).to(tl.float32)
        scale = scale * pw

    # Handle scale_grad_by_freq
    if scale_grad_by_freq:
        # Count occurrences of this index in offset2bag
        n_samples = tl.num_programs(0)
        cnt = 0
        for i in range(n_samples):
            loaded_idx = tl.load(indices_ptr + i).to(tl.int32)
            cnt = cnt + (loaded_idx == idx)
        if cnt > 1:
            scale = scale / tl.cast(cnt, tl.float32)

    # Apply scale to gradient
    go = go * scale

    # Store to output using atomic add
    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)


@libentry()
@triton.jit
def _embedding_bag_dense_backward_kernel_v2(
    grad_ptr,
    indices_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    mode,
    has_per_sample_weights,
    per_sample_weights_ptr,
    scale_grad_by_freq,
    num_samples,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    """Optimized kernel that avoids redundant loads.

    Args:
        grad_ptr: gradient tensor of shape (num_bags, embedding_dim)
        indices_ptr: indices into embedding table, shape (num_samples,)
        offset2bag_ptr: maps sample position to bag index, shape (num_samples,)
        bag_size_ptr: number of samples in each bag, shape (num_bags,)
        grad_weight_ptr: output gradient for embedding table, shape (num_weights, embedding_dim)
        num_weights: number of rows in embedding table
        padding_idx: padding index to ignore
        mode: 0=mean, 1=sum
        has_per_sample_weights: whether per_sample_weights are provided
        per_sample_weights_ptr: optional per-sample weights
        scale_grad_by_freq: whether to scale by word frequency
        num_samples: total number of samples
        BLOCK_D: block size for embedding dimension
        EMBED_DIM: embedding dimension
    """
    pid = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    # Load index and bag for this sample
    idx = tl.load(indices_ptr + pid).to(tl.int32)
    bag_idx = tl.load(offset2bag_ptr + pid).to(tl.int32)

    # Check validity
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    # Load bag size
    bag_size = tl.load(bag_size_ptr + bag_idx).to(tl.float32)
    bag_size = tl.where(bag_size == 0.0, 1.0, bag_size)

    # Load gradient for this bag: load all embedding_dim values at once
    base_go = bag_idx * EMBED_DIM
    go_ptrs = grad_ptr + base_go + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    # Compute scale factor
    if mode == 0:
        # Mean mode: divide by bag size
        scale = 1.0 / bag_size
    else:
        # Sum mode (mode=1): no scaling
        scale = 1.0

    # Handle per_sample_weights if provided
    if has_per_sample_weights:
        pw = tl.load(per_sample_weights_ptr + pid).to(tl.float32)
        scale = scale * pw

    # Handle scale_grad_by_freq
    if scale_grad_by_freq:
        # Count occurrences of this index in offset2bag
        cnt = 0
        for i in range(num_samples):
            loaded_idx = tl.load(indices_ptr + i).to(tl.int32)
            cnt = cnt + (loaded_idx == idx)
        if cnt > 1:
            scale = scale / tl.cast(cnt, tl.float32)

    # Apply scale to gradient
    go = go * scale

    # Store to output using atomic add
    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)


def _embedding_bag_dense_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    maximum_indices: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool,
    mode: int,
    per_sample_weights: torch.Tensor = None,
    padding_idx: int = -1,
) -> torch.Tensor:
    """Compute gradient for embedding_bag dense backward.

    Args:
        grad: gradient tensor of shape (num_bags, embedding_dim)
        indices: indices into embedding table of shape (num_samples,)
        offset2bag: maps each sample position to its bag of shape (num_samples,)
        bag_size: number of samples in each bag of shape (num_bags,)
        maximum_indices: indices of maximum values (unused for mode=0)
        num_weights: number of rows in embedding table
        scale_grad_by_freq: whether to scale gradients by word frequency
        mode: embedding bag mode (0=mean, 1=sum)
        per_sample_weights: optional per-sample weights of shape (num_samples,)
        padding_idx: padding index to ignore

    Returns:
        Gradient tensor of shape (num_weights, embedding_dim)
    """
    logger.debug("GEMS _embedding_bag_dense_backward")

    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    assert (
        grad.is_cuda
        and indices.is_cuda
        and offset2bag.is_cuda
        and bag_size.is_cuda
        and grad.device == indices.device
    ), "All inputs must be CUDA tensors on the same device."

    device = grad.device
    num_bags = grad.shape[0]
    D = grad.shape[1]
    num_samples = indices.numel()

    assert num_bags == bag_size.numel(), "num_bags must match bag_size length"
    assert num_samples == offset2bag.numel(), "num_samples must match offset2bag length"

    # Initialize output gradient
    grad_weight = torch.zeros(
        (num_weights, D), device=device, dtype=torch.float32
    ).contiguous()

    BLOCK_D = 128
    grid = (num_samples, triton.cdiv(D, BLOCK_D))

    # Create a dummy tensor for per_sample_weights if not provided
    # This ensures we always pass a valid tensor pointer to the kernel
    if per_sample_weights is not None:
        per_sample_weights_input = per_sample_weights.contiguous()
    else:
        per_sample_weights_input = torch.ones(
            num_samples, device=device, dtype=torch.float32
        )

    # Determine which kernel to use
    # For simplicity, use the basic version
    _embedding_bag_dense_backward_kernel[grid](
        grad,
        indices,
        offset2bag,
        bag_size,
        grad_weight,
        num_weights,
        padding_idx,
        mode,
        per_sample_weights is not None,
        per_sample_weights_input,
        scale_grad_by_freq,
        BLOCK_D=BLOCK_D,
        EMBED_DIM=D,
    )

    if grad.dtype != torch.float32:
        return grad_weight.to(grad.dtype)
    return grad_weight
