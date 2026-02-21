import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024


@triton.jit
def index_fill_kernel(
    out_ptr,
    index_ptr,
    value,
    outer_size,
    dim_size,
    inner_size,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = outer_size * M * inner_size
    mask = idx < total

    inner_idx = idx % inner_size
    m = (idx // inner_size) % M
    outer_idx = idx // (inner_size * M)

    dim_val = tl.load(index_ptr + m, mask=mask, other=0).to(tl.int64)
    out_offset = outer_idx * dim_size * inner_size + dim_val * inner_size + inner_idx
    tl.store(out_ptr + out_offset, value, mask=mask)


def _index_fill_impl(out: torch.Tensor, dim: int, index: torch.Tensor, value) -> None:
    shape = list(out.shape)
    dim_size = shape[dim]

    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]
    inner_size = 1
    for i in range(dim + 1, out.ndim):
        inner_size *= shape[i]

    M = index.size(0)
    total = outer_size * M * inner_size
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    # Cast value to a Python scalar compatible with the tensor dtype
    if out.is_floating_point():
        value = float(value)
    else:
        value = int(value)

    with torch_device_fn.device(out.device):
        index_fill_kernel[grid](
            out,
            index,
            value,
            outer_size,
            dim_size,
            inner_size,
            M,
            BLOCK_SIZE=BLOCK_SIZE,
        )


def index_fill(inp: torch.Tensor, dim: int, index: torch.Tensor, value) -> torch.Tensor:
    logger.debug("GEMS INDEX FILL")
    assert index.ndim == 1, "index must be a 1D tensor"
    assert -inp.ndim <= dim < inp.ndim, "dim out of range"
    dim = dim % inp.ndim

    out = inp.clone()
    M = index.size(0)
    if M == 0:
        return out

    _index_fill_impl(out, dim, index, value)
    return out


def index_fill_(
    inp: torch.Tensor, dim: int, index: torch.Tensor, value
) -> torch.Tensor:
    logger.debug("GEMS INDEX FILL_")
    assert index.ndim == 1, "index must be a 1D tensor"
    assert -inp.ndim <= dim < inp.ndim, "dim out of range"
    dim = dim % inp.ndim

    M = index.size(0)
    if M == 0:
        return inp

    _index_fill_impl(inp, dim, index, value)
    return inp
