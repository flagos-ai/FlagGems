import logging

import triton
import triton.language as tl

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def scatter_slice_kernel(
    out_ptr,
    src_ptr,
    src_elements,
    dim_prod_post,
    out_stride_dim,
    index_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < src_elements

    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post

    out_idx = pre_idx * out_stride_dim + index_offset + post_idx

    src_data = tl.load(src_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, src_data, mask=mask)


def select_scatter(inp, src, dim, index):
    logger.debug("GEMS_KUNLUNXIN SELECT_SCATTER")
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
        out = inp.clone()
    else:
        out = inp.clone()

    src = src.contiguous()
    out_contig = out.contiguous()

    src_elements = src.numel()
    if src_elements == 0:
        return out

    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    out_stride_dim = inp.size(dim) * dim_prod_post
    out_offset = index * dim_prod_post

    BLOCK_SIZE = 1024
    if src_elements >= 1024 * 1024:
        BLOCK_SIZE = 4096
    elif src_elements >= 4096:
        BLOCK_SIZE = 2048

    grid = (triton.cdiv(src_elements, BLOCK_SIZE),)

    scatter_slice_kernel[grid](
        out_contig,
        src,
        src_elements,
        dim_prod_post,
        out_stride_dim,
        out_offset,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_contig
