import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_select_backward import (
    index_select_backward as generic_index_select_backward,
)
from flag_gems.utils import dim_compress, libentry

from .mm import mm

logger = logging.getLogger(__name__)

_MAX_ONE_HOT_ELEMENTS = 20_000_000


@libentry()
@triton.jit
def _make_one_hot_kernel(
    out_ptr,
    index_ptr,
    index_len,
    dim_size_out,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = index_len * dim_size_out
    mask = offsets < total
    row = offsets // dim_size_out
    col = offsets % dim_size_out
    target = tl.load(index_ptr + row, mask=mask, other=-1)
    tl.store(out_ptr + offsets, (col == target).to(tl.float32), mask=mask)


def index_select_backward(grad, self_sizes, dim, index):
    logger.debug("GEMS_KUNLUNXIN INDEX_SELECT_BACKWARD")

    dim = dim % grad.ndim
    index_len = index.numel()
    dim_size_out = self_sizes[dim]
    one_hot_elements = index_len * dim_size_out

    if (
        index_len == 0
        or one_hot_elements > _MAX_ONE_HOT_ELEMENTS
        or grad.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return generic_index_select_backward(grad, self_sizes, dim, index)

    orig_dtype = grad.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        grad = grad.to(torch.float32)

    grad_compressed = dim_compress(grad, dim)
    grad_flat = grad_compressed.reshape(-1, index_len)

    one_hot = torch.empty(
        (index_len, dim_size_out),
        dtype=torch.float32,
        device=grad.device,
    )
    grid = (triton.cdiv(one_hot_elements, 1024),)
    _make_one_hot_kernel[grid](
        one_hot,
        index,
        index_len,
        dim_size_out,
        BLOCK=1024,
    )
    out_flat = mm(grad_flat, one_hot)

    compressed_shape = list(grad_compressed.shape)
    compressed_shape[-1] = dim_size_out
    out_flat = out_flat.reshape(compressed_shape)
    if dim != grad.ndim - 1:
        order = [i for i in range(out_flat.ndim - 1)]
        order.insert(dim, out_flat.ndim - 1)
        out = out_flat.permute(order).contiguous()
    else:
        out = out_flat

    if orig_dtype in (torch.float16, torch.bfloat16):
        return out.to(orig_dtype)
    return out
