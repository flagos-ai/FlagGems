import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024


@triton.jit
def diff_kernel(
    inp_ptr,
    out_ptr,
    outer_size,
    dim_size_in,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dim_size_out = dim_size_in - 1
    total = outer_size * dim_size_out * inner_size
    mask = idx < total

    inner_idx = idx % inner_size
    dim_idx = (idx // inner_size) % dim_size_out
    outer_idx = idx // (inner_size * dim_size_out)

    base = outer_idx * dim_size_in * inner_size + dim_idx * inner_size + inner_idx
    val0 = tl.load(inp_ptr + base, mask=mask, other=0.0)
    val1 = tl.load(inp_ptr + base + inner_size, mask=mask, other=0.0)

    tl.store(out_ptr + idx, val1 - val0, mask=mask)


def _diff_once(input: torch.Tensor, dim: int) -> torch.Tensor:
    ndim = input.ndim
    shape = list(input.shape)
    S = shape[dim]

    out_shape = shape[:]
    out_shape[dim] = S - 1

    if S <= 1:
        return torch.empty(out_shape, dtype=input.dtype, device=input.device)

    input = input.contiguous()
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]

    total = outer_size * (S - 1) * inner_size
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        diff_kernel[grid](
            input,
            out,
            outer_size,
            S,
            inner_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out


def diff(
    input: torch.Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[torch.Tensor] = None,
    append: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS DIFF")
    assert n >= 0, "n must be a non-negative integer"
    ndim = input.ndim
    assert -ndim <= dim < ndim, "dim out of range"
    if dim < 0:
        dim = ndim + dim

    if prepend is not None or append is not None:
        parts = []
        if prepend is not None:
            parts.append(prepend)
        parts.append(input)
        if append is not None:
            parts.append(append)
        input = torch.cat(parts, dim=dim)

    if n == 0:
        return input.clone()

    for _ in range(n):
        if input.size(dim) == 0:
            break
        input = _diff_once(input, dim)

    return input
