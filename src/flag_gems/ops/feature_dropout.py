import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def generate_feature_mask_kernel(
    MASK,
    N,  # batch size
    C,  # number of channels
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Generate a feature dropout mask of shape (N, C).
    Each element is either 0 (dropped) or scale (kept).
    Each (n, c) pair gets its own random value.
    """
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    n_mask = n_offset < N
    c_mask = c_offset < C

    # Compute flat indices for random number generation
    # flat_idx = n * C + c
    flat_idx = n_offset[:, None] * C + c_offset[None, :]

    # Generate random numbers using philox
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = flat_idx.to(tl.uint32)
    c0 = c0 + i4
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    rand_vals = uint_to_uniform_float(r0)

    # Create mask: scale if rand > p (keep), 0 if rand <= p (drop)
    mask_vals = tl.where(rand_vals > p, scale, 0.0)

    # Store mask
    mask_offsets = n_offset[:, None] * C + c_offset[None, :]
    mask_mask = n_mask[:, None] & c_mask[None, :]
    tl.store(MASK + mask_offsets, mask_vals, mask=mask_mask)


@triton.jit
def apply_feature_mask_kernel(
    X,
    Y,
    MASK,
    numel,
    N,  # batch size
    C,  # channels
    spatial_size,  # H * W or D1 * D2 * ...
    BLOCK: tl.constexpr,
):
    """
    Apply feature mask to input tensor.
    Input shape: (N, C, ...) flattened to (numel,)
    Mask shape: (N, C)

    For element at flat index i:
    - For contiguous (N, C, H, W) layout: i = n * (C * spatial) + c * spatial + spatial_idx
    - n = i // (C * spatial_size)
    - c = (i // spatial_size) % C
    - mask_idx = n * C + c
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < numel

    # Compute batch and channel index for each element
    channel_spatial_size = C * spatial_size
    n_idx = offset // channel_spatial_size
    c_idx = (offset % channel_spatial_size) // spatial_size

    # Compute mask index: n * C + c
    mask_idx = n_idx * C + c_idx

    # Load input and mask
    x = tl.load(X + offset, mask=mask, other=0.0)
    m = tl.load(MASK + mask_idx, mask=mask, other=0.0)

    # Apply mask
    y = x * m

    tl.store(Y + offset, y, mask=mask)


def feature_dropout(input, p, train=True):
    """
    Applies feature dropout to the input tensor.

    Randomly zeroes out entire channels of the input tensor with probability p.
    Each batch element has its own independent channel mask.

    Args:
        input: Input tensor of shape (N, C, ...) where N is batch size, C is channels
        p: Probability of a channel to be zeroed. Default: 0.5
        train: If True, applies dropout. If False, returns input unchanged.

    Returns:
        Output tensor of same shape as input
    """
    logger.debug("GEMS FEATURE_DROPOUT")

    if not train or p == 0:
        return input.clone()

    if p == 1:
        return torch.zeros_like(input)

    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)

    # Get dimensions
    batch_size = input.shape[0]
    num_channels = input.shape[1]
    spatial_size = 1
    for i in range(2, input.ndim):
        spatial_size *= input.shape[i]

    N = batch_size
    C = num_channels
    numel = input.numel()
    scale = 1.0 / (1.0 - p)

    # Create mask tensor of shape (N, C)
    mask = torch.empty(N, C, device=device, dtype=torch.float32)

    # Generate mask
    BLOCK_N = min(triton.next_power_of_2(N), 64)
    BLOCK_C = min(triton.next_power_of_2(C), 64)
    grid_mask = (triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C))

    # Need N * C random numbers
    increment = triton.cdiv(N * C, 4) * 4
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        generate_feature_mask_kernel[grid_mask](
            mask, N, C, p, scale, philox_seed, philox_offset, BLOCK_N, BLOCK_C
        )

    # Apply mask to input
    BLOCK = 1024
    grid_apply = (triton.cdiv(numel, BLOCK),)

    with torch_device_fn.device(device):
        apply_feature_mask_kernel[grid_apply](
            input, out, mask, numel, N, C, spatial_size, BLOCK
        )

    return out


def feature_dropout_(input, p, train=True):
    """
    In-place version of feature_dropout.
    """
    logger.debug("GEMS FEATURE_DROPOUT_")
    if not train or p == 0:
        return input
    if p == 1:
        input.zero_()
        return input
    out = feature_dropout(input, p, train)
    input.copy_(out)
    return input
