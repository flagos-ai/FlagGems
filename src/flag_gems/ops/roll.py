import logging
from typing import List, Optional, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def roll_kernel(
    inp_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    # For a flat roll (no dims specified), just shift by flat_shift
    flat_shift,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    # Compute source index: (offset - flat_shift) % N
    src_idx = (offset - flat_shift + N) % N

    val = tl.load(inp_ptr + src_idx, mask=mask)
    tl.store(out_ptr + offset, val, mask=mask)


def _normalize_shifts_dims(shifts, dims):
    """Normalize shifts and dims to lists."""
    if isinstance(shifts, int):
        shifts = [shifts]
    else:
        shifts = list(shifts)

    if dims is None:
        dims = None
    elif isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    return shifts, dims


def roll(
    inp: torch.Tensor,
    shifts: Union[int, List[int]],
    dims: Optional[Union[int, List[int]]] = None,
) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    shifts, dims = _normalize_shifts_dims(shifts, dims)

    if dims is None or len(dims) == 0:
        # Flat roll: flatten, shift, reshape
        flat = inp.contiguous().view(-1)
        N = flat.numel()
        if N == 0:
            return inp.clone()

        total_shift = sum(shifts) % N
        if total_shift == 0:
            return inp.clone()

        out_flat = torch.empty_like(flat)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)

        with torch_device_fn.device(inp.device):
            roll_kernel[grid](flat, out_flat, N, BLOCK_SIZE, total_shift)

        return out_flat.view(inp.shape)
    else:
        # Dimension-wise roll: apply shifts sequentially per dim
        assert len(shifts) == len(dims), (
            f"shifts and dims must have the same length, "
            f"got {len(shifts)} and {len(dims)}"
        )

        result = inp
        for shift, dim in zip(shifts, dims):
            dim = dim % result.ndim
            size = result.size(dim)
            if size == 0:
                continue
            shift = shift % size
            if shift == 0:
                continue
            # Use torch.cat with narrow for each dimension roll
            # This is efficient and works with Triton since the underlying
            # ops (narrow, cat) are already optimized
            result = torch.cat(
                (
                    result.narrow(dim, size - shift, shift),
                    result.narrow(dim, 0, size - shift),
                ),
                dim=dim,
            )

        return result.contiguous()
