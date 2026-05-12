import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("nonzero"),
    key=[
        "n_elements",
    ],
)
@triton.jit
def nonzero_kernel(
    inp,
    prefix_sum,
    out,
    n_elements,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offset = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements

        inp_vals = tl.load(inp + offset, mask=mask, other=0.0).to(tl.int1)
        nonzero_mask = mask and inp_vals
        out_row_offset = tl.load(prefix_sum + offset, mask=nonzero_mask) - 1
        out_col_offset = tl.arange(0, ndim)
        out_offsets = out_row_offset[:, None] * ndim + out_col_offset[None, :]
        out_vals = tl.zeros((BLOCK_SIZE, ndim), tl.int32)
        idx_flat = offset
        for dim in range(ndim - 1, -1, -1):
            dim_size = tl.load(shape + dim)
            remainder = idx_flat % dim_size
            idx_flat //= dim_size
            out_vals[:, dim] = remainder
        tl.store(out + out_offsets.to(tl.int32), out_vals, mask=nonzero_mask[:, None])


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS_CAMBRICON NONZERO")

    inp_ndim = inp.ndim

    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)

    shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)

    inp_bool = inp_view
    if inp_view.dtype != torch.bool:
        inp_bool = inp_view != 0

    prefix_sum = inp_bool.cumsum(axis=0)
    num_nonzeros = n_elements
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)
    grid = lambda meta: (
        min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
    )
    with torch_device_fn.device(inp.device):
        nonzero_kernel[grid](inp_bool, prefix_sum, out, n_elements, shape, inp_ndim)

    num_nonzeros = prefix_sum[n_elements - 1].item()
    out = out[0:num_nonzeros]

    if as_tuple:
        return torch.unbind(out, dim=0)
    else:
        return out
