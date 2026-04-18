import logging
from typing import List, Optional, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _roll_kernel(
    in_ptr,
    out_ptr,
    numel,
    # Per-dim info passed as flat arrays
    shape_ptr,
    strides_ptr,
    shifts_ptr,
    dims_ptr,
    ndim: tl.constexpr,
    nroll: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    # For each output linear index, compute the corresponding input linear index
    # by reversing the roll shift on each rolled dimension.
    linear = offsets

    # Decompose linear index into per-dim indices, apply inverse shift, recompose
    src_linear = tl.zeros_like(linear)
    remaining = linear
    for d in tl.static_range(ndim):
        dim_size = tl.load(shape_ptr + (ndim - 1 - d))
        dim_stride = tl.load(strides_ptr + (ndim - 1 - d))
        idx = remaining % dim_size
        remaining = remaining // dim_size

        # Check if this dimension is rolled
        shift = tl.zeros_like(idx)
        for r in tl.static_range(nroll):
            roll_dim = tl.load(dims_ptr + r)
            roll_shift = tl.load(shifts_ptr + r)
            is_this_dim = (roll_dim == (ndim - 1 - d))
            shift = tl.where(is_this_dim, roll_shift, shift)

        # Inverse shift: src_idx = (out_idx - shift) % size
        src_idx = (idx - shift % dim_size + dim_size) % dim_size
        src_linear = src_linear + src_idx * dim_stride

    val = tl.load(in_ptr + src_linear, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


def roll(
    input: torch.Tensor,
    shifts: Union[int, List[int]],
    dims: Optional[Union[int, List[int]]] = None,
) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    if dims is None:
        # Flatten, roll, reshape
        flat = input.contiguous().view(-1)
        result = roll(flat, shifts, dims=0)
        return result.view(input.shape)

    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]

    ndim = input.ndim
    # Normalize negative dims
    dims = [d % ndim for d in dims]
    # Normalize shifts
    shifts = [s % input.shape[d] if input.shape[d] > 0 else 0 for s, d in zip(shifts, dims)]

    input_c = input.contiguous()
    out = torch.empty_like(input_c)
    numel = input_c.numel()

    if numel == 0:
        return out

    shape = list(input_c.shape)
    strides = list(input_c.stride())
    nroll = len(shifts)

    shape_t = torch.tensor(shape, dtype=torch.int64, device=input.device)
    strides_t = torch.tensor(strides, dtype=torch.int64, device=input.device)
    shifts_t = torch.tensor(shifts, dtype=torch.int64, device=input.device)
    dims_t = torch.tensor(dims, dtype=torch.int64, device=input.device)

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)

    with torch_device_fn.device(input.device):
        _roll_kernel[grid](
            input_c,
            out,
            numel,
            shape_t,
            strides_t,
            shifts_t,
            dims_t,
            ndim=ndim,
            nroll=nroll,
            BLOCK=BLOCK,
        )
    return out
