import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("elementwise_generic"))
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
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask).to(tl.int1)
    out_offset = tl.load(prefix_sum + offset, mask=mask) - 1

    nonzero_mask = mask and inp_vals  # noqa

    idx_flat = offset
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + out_offset * ndim + dim, remainder, mask=nonzero_mask)


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS NONZERO")

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

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(inp.device):
        nonzero_kernel[grid](inp_bool, prefix_sum, out, n_elements, shape, inp_ndim)

    num_nonzeros = prefix_sum[n_elements - 1].item()
    out = out[0:num_nonzeros]

    if as_tuple:
        return torch.unbind(out, dim=0)
    else:
        return out
