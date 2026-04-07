import logging
from typing import List, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Triton kernel for circular shift along a single dimension.
# Computes shifted index for each element and copies data.
@triton.jit
def roll_kernel(
    inp_ptr,
    out_ptr,
    total_elements,
    shift,
    dim_size,
    dim_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute the index along the roll dimension
    dim_idx = (offsets // dim_stride) % dim_size
    # Apply shift with wrapping
    new_dim_idx = (dim_idx + shift) % dim_size
    # Compute output index
    out_offsets = offsets + (new_dim_idx - dim_idx) * dim_stride

    val = tl.load(inp_ptr + offsets, mask=mask)
    tl.store(out_ptr + out_offsets, val, mask=mask)


def roll(
    A: torch.Tensor,
    shifts: Union[int, List[int]],
    dims: Union[int, List[int]] = None,
) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    if isinstance(shifts, int):
        shifts = [shifts]
    if dims is None:
        # When dims is None, roll the flattened tensor then reshape
        A_flat = A.contiguous().view(-1)
        shift_val = shifts[0] % A_flat.numel() if A_flat.numel() > 0 else 0
        out_flat = torch.empty_like(A_flat)
        if A_flat.numel() == 0:
            return out_flat.view(A.shape)
        total = A_flat.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total, BLOCK_SIZE),)
        with torch_device_fn.device(A.device):
            roll_kernel[grid](
                A_flat, out_flat, total, shift_val, total, 1, BLOCK_SIZE=BLOCK_SIZE
            )
        return out_flat.view(A.shape)
    else:
        if isinstance(dims, int):
            dims = [dims]
        assert len(shifts) == len(dims), "shifts and dims must have the same length"

        A = A.contiguous()
        out = A
        for shift_val, dim in zip(shifts, dims):
            dim = dim % A.ndim
            dim_size = out.shape[dim]
            if dim_size == 0:
                continue
            shift_val = shift_val % dim_size
            if shift_val == 0:
                continue

            inp = out.contiguous()
            out = torch.empty_like(inp)
            total = inp.numel()
            # stride of the dimension being rolled
            dim_stride = 1
            for d in range(A.ndim - 1, dim, -1):
                dim_stride *= inp.shape[d]

            BLOCK_SIZE = 1024
            grid = (triton.cdiv(total, BLOCK_SIZE),)
            with torch_device_fn.device(A.device):
                roll_kernel[grid](
                    inp,
                    out,
                    total,
                    shift_val,
                    dim_size,
                    dim_stride,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
        return out
