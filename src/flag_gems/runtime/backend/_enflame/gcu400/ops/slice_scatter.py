import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=['total_elements'])
def slice_scatter_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    total_elements,
    dim_size,
    start,
    step,
    src_dim_size,
    BLOCK: tl.constexpr,
    DPP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elements

    inner = dim_size * DPP
    pre_idx = offs // inner
    rem = offs - pre_idx * inner

    if DPP == 1:
        dim_idx = rem
        post_idx = tl.zeros_like(rem)
    else:
        dim_idx = rem // DPP
        post_idx = rem - dim_idx * DPP

    inp_data = tl.load(inp_ptr + offs, mask=mask)

    end_val = start + src_dim_size * step
    slice_mask = (
        (dim_idx >= start)
        & (dim_idx < end_val)
        & ((dim_idx - start) % step == 0)
    )

    src_dim_idx = (dim_idx - start) // step
    src_off = pre_idx * src_dim_size * DPP + src_dim_idx * DPP + post_idx
    src_data = tl.load(src_ptr + src_off, mask=mask & slice_mask)
    result = tl.where(slice_mask, src_data, inp_data)
    tl.store(out_ptr + offs, result, mask=mask)


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

    dim_size = inp.size(dim)

    valid_shape = list(inp.shape)
    valid_shape[dim] = triton.cdiv(end - start, step)
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        inp = inp.contiguous()

    total_elements = inp.numel()

    if total_elements >= 131072:
        if step == 1:
            out = torch.empty_like(inp)
            if start > 0:
                out.narrow(dim, 0, start).copy_(inp.narrow(dim, 0, start))
            src_dim_size = src.size(dim)
            out.narrow(dim, start, src_dim_size).copy_(src)
            tail = dim_size - start - src_dim_size
            if tail > 0:
                out.narrow(dim, start + src_dim_size, tail).copy_(
                    inp.narrow(dim, start + src_dim_size, tail)
                )
            return out
        out = inp.clone()
        slices = [slice(None)] * inp.ndim
        slices[dim] = slice(start, end, step)
        out[tuple(slices)].copy_(src)
        return out

    out = torch.empty_strided(
        inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
    )
    inp = inp.contiguous()
    src = src.contiguous()

    src_dim_size = src.size(dim)

    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    BLOCK = 1024
    grid = (triton.cdiv(total_elements, BLOCK),)

    slice_scatter_kernel[grid](
        out, inp, src,
        total_elements, dim_size,
        start, step, src_dim_size,
        BLOCK=BLOCK, DPP=dim_prod_post,
        num_warps=4,
    )

    return out
