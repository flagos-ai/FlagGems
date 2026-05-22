import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def histc_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    bins: tl.constexpr,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute histogram of input tensor.
    Each thread processes BLOCK_SIZE elements, computing which bin they belong to
    and atomically incrementing the corresponding bin counter.
    """
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=0.0)

    # Convert to float32 for computation
    inp_val = inp_val.to(tl.float32)

    # Compute bin range
    bin_width = (max_val - min_val) / bins

    # Compute bin indices
    # Elements equal to max_val go to the last bin (bins - 1)
    # Elements outside [min_val, max_val] or NaN are ignored
    bin_idx = ((inp_val - min_val) / bin_width).to(tl.int32)

    # Clamp to valid range [0, bins-1] for elements in range
    # Elements outside range or NaN should be excluded
    in_range = (inp_val >= min_val) & (inp_val <= max_val)

    # Handle edge case: elements exactly equal to max go to last bin
    bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)

    # Only count elements in range (excludes NaN via the comparison)
    valid_mask = mask & in_range

    # Atomic add to histogram bins
    # We need to iterate through each element and add to the appropriate bin
    for i in range(BLOCK_SIZE):
        if tl.load(valid_mask.to(tl.int8).reshape(BLOCK_SIZE) + i) != 0:
            idx = tl.load(bin_idx.reshape(BLOCK_SIZE) + i)
            tl.atomic_add(out_ptr + idx, 1.0, sem="relaxed")


@libentry()
@triton.jit
def histc_kernel_simple(
    inp_ptr,
    out_ptr,
    n_elements,
    bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple histogram kernel - each program handles one element at a time.
    """
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load input values
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=float("nan"))

    # Convert to float32 for computation
    inp_val = inp_val.to(tl.float32)

    # Compute bin indices using multiplication to avoid float division precision loss
    bin_idx = tl.floor((inp_val - min_val) * bins / (max_val - min_val)).to(tl.int64)

    # Handle edge case: elements exactly equal to max go to last bin
    bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)

    # Check if elements are in valid range (excludes NaN)
    in_range = (inp_val >= min_val) & (inp_val <= max_val)

    # Clamp bin indices to valid range
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)

    valid_mask = mask & in_range

    # Atomically add to histogram
    tl.atomic_add(out_ptr + bin_idx, 1.0, mask=valid_mask, sem="relaxed")


def histc(inp, bins=100, min=0, max=0):
    """
    Compute the histogram of a tensor.

    Args:
        inp: Input tensor
        bins: Number of histogram bins (default: 100)
        min: Lower end of the range (inclusive). If min == max == 0, uses data min.
        max: Upper end of the range (inclusive). If min == max == 0, uses data max.

    Returns:
        Tensor: Histogram represented as a tensor of shape (bins,)
    """
    logger.debug("GEMS HISTC")

    # Ensure input is contiguous
    inp = inp.contiguous()

    # Get min and max values
    min_val = float(min)
    max_val = float(max)

    if min_val == 0 and max_val == 0:
        # Use actual min/max of the data
        min_val = float(inp.min().item())
        max_val = float(inp.max().item())

    # Handle edge case where min == max
    if min_val == max_val:
        # All elements go to the first bin if they equal min_val
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        # Count how many elements equal min_val (excluding NaN)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        out[0] = count
        return out

    # Create output histogram tensor
    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()

    if n_elements == 0:
        return out

    # Choose block size
    BLOCK_SIZE = 1024

    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(inp.device):
        histc_kernel_simple[grid](
            inp,
            out,
            n_elements,
            bins,
            min_val,
            max_val,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out
