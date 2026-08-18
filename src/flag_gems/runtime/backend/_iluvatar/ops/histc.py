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
    """Histogram kernel with vectorized atomic_add for Iluvatar BI-V150.

    Each program processes BLOCK_SIZE consecutive elements and uses
    tl.atomic_add with a validity mask for efficient scatter into
    the output histogram bins.
    """
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=float("nan"))
    inp_val = inp_val.to(tl.float32)
    bin_width = (max_val - min_val) / bins
    bin_idx = ((inp_val - min_val) / bin_width).to(tl.int64)
    bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
    in_range = (inp_val >= min_val) & (inp_val <= max_val)
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)
    valid_mask = mask & in_range
    tl.atomic_add(out_ptr + bin_idx, 1.0, mask=valid_mask, sem="relaxed")


def histc(inp, bins=100, min=0, max=0):
    """Compute the histogram of a tensor.

    Iluvatar BI-V150 backend implementation using a libentry-decorated
    Triton kernel with vectorized atomic_add.

    Args:
        inp: Input tensor.
        bins: Number of histogram bins (default: 100).
        min: Lower end of the range (inclusive).
        max: Upper end of the range (inclusive).

    Returns:
        Histogram tensor of shape (bins,).
    """
    logger.debug("GEMS HISTC (Iluvatar)")
    inp = inp.contiguous()
    min_val = float(min)
    max_val = float(max)
    if min_val == 0 and max_val == 0:
        min_val = float(inp.min().item())
        max_val = float(inp.max().item())
    if min_val == max_val:
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        out[0] = count
        return out
    n_elements = inp.numel()
    if n_elements == 0:
        return torch.zeros(bins, dtype=inp.dtype, device=inp.device)

    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(inp.device):
        histc_kernel[grid](
            inp,
            out,
            n_elements,
            bins,
            min_val,
            max_val,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out
