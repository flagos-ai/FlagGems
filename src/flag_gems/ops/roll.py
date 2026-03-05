import logging
from typing import List, Optional, Union

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def roll_kernel(
    X,
    Y,
    total_n,
    roll_size,
    shift,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)

    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_n

    # Decompose linear index into (outer_idx, roll_idx, inner_idx)
    # But for simplicity, we compute flat source index
    # The rolled dimension has size roll_size
    # inner_size is folded into total_n as: total_n = outer_size * roll_size * inner_size
    # We use the flat approach: each block handles a contiguous chunk of the output

    batch_offset = pid_batch * total_n
    out_idx = offsets

    # src_idx = (out_idx - shift) % total_n  -- only works for flat roll (no dims)
    # For dim-specific roll, we need to decompose
    # This kernel is for the flat (no dims) case
    src_idx = (out_idx + total_n - shift % total_n) % total_n

    x = tl.load(X + batch_offset + src_idx, mask=mask)
    tl.store(Y + batch_offset + out_idx, x, mask=mask)


@libentry()
@triton.jit
def roll_dim_kernel(
    X,
    Y,
    inner_size,
    roll_size,
    shift,
    stride_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_outer = tl.program_id(1)

    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    n_inner_roll = roll_size * inner_size
    mask = offsets < n_inner_roll

    # Decompose into (roll_idx, inner_idx)
    roll_idx = offsets // inner_size
    inner_idx = offsets % inner_size

    # Source roll index with circular shift
    src_roll_idx = (roll_idx + roll_size - shift % roll_size) % roll_size

    out_offset = pid_outer * n_inner_roll + roll_idx * inner_size + inner_idx
    src_offset = pid_outer * n_inner_roll + src_roll_idx * inner_size + inner_idx

    x = tl.load(X + src_offset, mask=mask)
    tl.store(Y + out_offset, x, mask=mask)


def roll(
    A: torch.Tensor,
    shifts: Union[int, List[int]],
    dims: Optional[Union[int, List[int]]] = None,
) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    if A.numel() == 0:
        return A.clone()

    if isinstance(shifts, int):
        shifts = [shifts]
    if dims is None:
        dims = []
    elif isinstance(dims, int):
        dims = [dims]

    assert len(shifts) == len(dims) or len(dims) == 0, (
        "shifts and dims must have the same length, or dims must be empty"
    )

    # Make input contiguous
    A = A.contiguous()

    if len(dims) == 0:
        # Flat roll: treat tensor as 1D
        total_n = A.numel()
        shift = shifts[0] % total_n
        if shift == 0:
            return A.clone()

        out = torch.empty_like(A)
        flat_A = A.view(-1)
        flat_out = out.view(-1)

        BLOCK = 1024
        grid = (triton.cdiv(total_n, BLOCK), 1)
        roll_kernel[grid](flat_A, flat_out, total_n, total_n, shift, BLOCK=BLOCK)
        return out
    else:
        # Dim-specific roll: apply shifts one by one
        result = A
        for shift, dim in zip(shifts, dims):
            dim = dim % A.ndim
            roll_size = result.shape[dim]
            shift = shift % roll_size
            if shift == 0:
                continue

            result = result.contiguous()
            out = torch.empty_like(result)

            # outer_size = product of dims before `dim`
            # inner_size = product of dims after `dim`
            outer_size = 1
            for i in range(dim):
                outer_size *= result.shape[i]
            inner_size = 1
            for i in range(dim + 1, result.ndim):
                inner_size *= result.shape[i]

            n_inner_roll = roll_size * inner_size
            BLOCK = 1024
            grid = (triton.cdiv(n_inner_roll, BLOCK), outer_size)

            roll_dim_kernel[grid](
                result, out, inner_size, roll_size, shift, n_inner_roll, BLOCK=BLOCK
            )
            result = out

        if result is A:
            return A.clone()
        return result
