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
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple flat copy — source indices are precomputed on CPU."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    x = tl.load(X + offsets, mask=mask)
    tl.store(Y + offsets, x, mask=mask)


def roll(
    self,
    shifts: Union[int, List[int]],
    dims: Optional[Union[int, List[int]]] = None,
):
    logger.debug("GEMS ROLL")

    if self.numel() == 0:
        return self.clone()

    if dims is None:
        # No dims specified: flatten, roll, reshape
        if isinstance(shifts, (list, tuple)):
            shift = shifts[0]
        else:
            shift = shifts
        n = self.numel()
        shift = shift % n if n > 0 else 0
        if shift == 0:
            return self.clone()

        # Use torch.cat on flattened — very fast, single kernel
        flat = self.reshape(-1)
        return torch.cat([flat[n - shift :], flat[: n - shift]]).reshape(self.shape)

    # Normalize to lists
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]

    assert len(shifts) == len(dims), (
        f"shifts and dims must have same length, " f"got {len(shifts)} and {len(dims)}"
    )

    result = self
    for shift, dim in zip(shifts, dims):
        actual_dim = dim % result.dim()
        dim_size = result.shape[actual_dim]
        if dim_size == 0:
            continue
        shift = shift % dim_size
        if shift == 0:
            continue

        # Use torch.cat with narrow — single kernel per dim, very fast
        result = torch.cat(
            [
                result.narrow(actual_dim, dim_size - shift, shift),
                result.narrow(actual_dim, 0, dim_size - shift),
            ],
            dim=actual_dim,
        )

    return result
