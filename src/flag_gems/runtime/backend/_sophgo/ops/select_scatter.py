import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
@libentry()
@triton.jit
def select_scatter_dim1_write_slice_kernel(
    src_ptr,
    out_ptr,
    OUTER,
    INNER,
    index,
    out_stride0,
    out_stride1,
    out_stride2,
    src_stride0,
    src_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    outer = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inner = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (outer < OUTER) & (inner < INNER)
    src_vals = tl.load(
        src_ptr + outer * src_stride0 + inner * src_stride1,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + outer * out_stride0 + index * out_stride1 + inner * out_stride2,
        src_vals,
        mask=mask,
    )


def select_scatter(inp, src, dim, index):
    logging.debug("GEMS SELECT_SCATTER SOPHGO")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)

    valid_shape = list(inp.shape)
    del valid_shape[dim]
    assert list(src.shape) == valid_shape, "Expected src to have a size equal to the slice of self"

    if inp.is_contiguous() and src.is_contiguous():
        outer = 1
        for size in inp.shape[:dim]:
            outer *= size
        inner = int(inp.numel() // (outer * inp.shape[dim]))
        out = inp.clone().reshape(outer, inp.shape[dim], inner)
        src_view = src.reshape(outer, inner)
        grid = (triton.cdiv(outer, 64), triton.cdiv(inner, 64))
        with torch_device_fn.device(inp.device):
            select_scatter_dim1_write_slice_kernel[grid](
                src_view,
                out,
                outer,
                inner,
                index,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                src_view.stride(0),
                src_view.stride(1),
                BLOCK_M=64,
                BLOCK_N=64,
            )
        return out.reshape_as(inp)

    # Keep the validated 2D dim=1 fused path. For the broader surface, clone
    # the full input first and then use indexed assignment for the selected
    # slice. This avoids the copy-on-view issues that appeared in the new
    # pytest coverage while keeping the historical hotspot optimization intact.
    out = inp.clone()
    out.narrow(dim, index, 1).copy_(src.unsqueeze(dim))
    return out
