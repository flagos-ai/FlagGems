import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)


@triton.jit
def select_scatter_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    total_elements,
    dim_size,
    dim_prod_post,
    index,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_elements
    idx = block_start + offsets

    pre_idx = idx // (dim_size * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size
    post_idx = idx % dim_prod_post

    select_mask = dim_idx == index

    inp_data = tl.load(inp_ptr + idx, mask=mask)

    src_idx = pre_idx * dim_prod_post + post_idx
    src_data = tl.load(src_ptr + src_idx, mask=mask & select_mask)
    result = tl.where(select_mask, src_data, inp_data)
    tl.store(out_ptr + idx, result, mask=mask)


def select_scatter(inp, src, dim, index):
    logger.debug("GEMS SELECT_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)

    valid_shape = list(inp.shape)
    del valid_shape[dim]
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    inp = inp.contiguous()
    src = src.contiguous()

    total_elements = inp.numel()
    dim_size = inp.size(dim)

    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    select_scatter_kernel[grid](
        out,
        inp,
        src,
        total_elements,
        dim_size,
        dim_prod_post,
        index,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
