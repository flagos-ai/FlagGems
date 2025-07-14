import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)


@triton.jit
def slice_scatter_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    total_elements,
    dim_size,
    dim_prod_post,
    start,
    step,
    src_dim_size,
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

    slice_mask = (
        (dim_idx >= start)
        & (dim_idx < start + src_dim_size * step)
        & ((dim_idx - start) % step == 0)
    )

    inp_data = tl.load(inp_ptr + idx, mask=mask)

    src_dim_idx = (dim_idx - start) // step
    src_idx = (
        pre_idx * src_dim_size * dim_prod_post + src_dim_idx * dim_prod_post + post_idx
    )
    src_data = tl.load(src_ptr + src_idx, mask=mask & slice_mask)
    result = tl.where(slice_mask, src_data, inp_data)
    tl.store(out_ptr + idx, result, mask=mask)


def slice_scatter(inp, src, dim=0, start=None, end=None, step=1):
    logger.debug("GEMS SLICE_SCATTER")
    assert src.device == inp.device, "inp and src reside on different devices."
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert step > 0, "slice step must be positive"
    dim = dim % inp.ndim

    start = start or 0
    end = end or inp.size(dim)
    if end < 0:
        end = end % inp.size(dim)

    valid_shape = list(inp.shape)
    valid_shape[dim] = triton.cdiv(end - start, step)
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
    src_dim_size = src.size(dim)

    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    slice_scatter_kernel[grid](
        out,
        inp,
        src,
        total_elements,
        dim_size,
        dim_prod_post,
        start,
        step,
        src_dim_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
