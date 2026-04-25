import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def roll_kernel(
    inp_ptr,
    out_ptr,
    N,
    dim_size,
    shift,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offset < N

    # Decompose flat index into (outer, dim_idx, inner)
    outer_stride = dim_size * inner_size
    outer_idx = offset // outer_stride
    remainder = offset % outer_stride
    dim_idx = remainder // inner_size
    inner_idx = remainder % inner_size

    # Apply roll: source_dim_idx = (dim_idx - shift) % dim_size
    source_dim_idx = (dim_idx - shift + dim_size) % dim_size

    # Reconstruct source flat index
    source_offset = outer_idx * outer_stride + source_dim_idx * inner_size + inner_idx

    val = tl.load(inp_ptr + source_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offset, val, mask=mask)


def _roll_single_dim(inp: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    size = inp.size(dim)
    if size == 0:
        return inp.clone()

    shift = shift % size
    if shift == 0:
        return inp.clone()

    inp_contig = inp.contiguous()
    out = torch.empty_like(inp_contig)

    inner_size = 1
    for i in range(dim + 1, inp.ndim):
        inner_size *= inp.size(i)

    N = inp.numel()
    BLOCK_SIZE = 512
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)

    roll_kernel[grid](
        inp_contig,
        out,
        N,
        size,
        shift,
        inner_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def roll(inp: torch.Tensor, shifts, dims=None) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    if inp.numel() == 0:
        return inp.clone()

    if dims is None:
        if isinstance(shifts, (list, tuple)):
            shift = shifts[0] if len(shifts) == 1 else sum(shifts)
        else:
            shift = shifts
        original_shape = inp.shape
        flat = inp.contiguous().reshape(-1)
        out_flat = _roll_single_dim(flat, shift, 0)
        return out_flat.reshape(original_shape)

    if isinstance(dims, int):
        dims = [dims]
    if isinstance(shifts, int):
        shifts = [shifts]

    assert len(shifts) == len(dims), "shifts and dims must have the same length"

    ndim = inp.ndim
    dims = [d % ndim for d in dims]

    result = inp
    for shift, dim in zip(shifts, dims):
        result = _roll_single_dim(result, shift, dim)
    return result
