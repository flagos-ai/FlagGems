import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val, argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
        in_val = tl.where(in_val.dtype.is_fp64(), in_val, in_val.to(tl.float32))
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val).to(tl.int32)
    index_val = tl.arange(0, BLOCK_SIZE)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )
    tl.store(out_ptr, sorted_in_val, mask=mask)
    tl.store(out_index_ptr, sorted_index_val, mask=mask)


def sort(inp, dim=-1, descending=False):
    logger.debug("GEMS_ASCEND SORT")
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)
    elif sort_elem_cnt > 128:  # TODO: Optimize implementation for large cases.
        return torch.sort(inp, stable=False, dim=dim, descending=descending)
    block_size = triton.next_power_of_2(sort_elem_cnt)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()
    batch_size = math.prod(inp.shape) // sort_elem_cnt

    out = torch.empty_like(inp)
    out_index = torch.empty_like(inp, dtype=torch.int64)

    with torch_device_fn.device(inp.device):
        sort_kernel[batch_size,](
            inp,
            out,
            out_index,
            N=sort_elem_cnt,
            BLOCK_SIZE=block_size,
            DESCENDING=descending,
            IS_FLOAT=inp.is_floating_point(),
        )

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
