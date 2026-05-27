import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_select"), key=["n_elements"])
@triton.jit
def masked_select_kernel(
    inp_ptr,
    select_mask_ptr,
    prefix_sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
    active = select_mask & mask
    inp_vals = tl.load(inp_ptr + offsets, mask=active, other=0)
    prefix_vals = tl.load(prefix_sum_ptr + offsets, mask=active, other=0)
    write_mask = active & (prefix_vals > 0)
    out_offset = prefix_vals - 1

    tl.store(out_ptr + out_offset, inp_vals, mask=write_mask)


def masked_select(inp, mask):
    logger.debug("GEMS MASKED SELECT")

    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    assert broadcastable(
        inp_shape, mask_shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"
    inp, mask = torch.broadcast_tensors(inp, mask)

    inp = inp.contiguous()
    mask = mask.contiguous()

    mask_flattened = mask.ravel()

    if mask_flattened.numel() <= 4096:
        prefix_sum = torch.cumsum(mask_flattened, dim=0, dtype=torch.int32)
    else:
        prefix_sum = torch.cumsum(mask_flattened.cpu(), dim=0, dtype=torch.int32).to(
            inp.device
        )
    out = torch.empty(prefix_sum[-1].item(), dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(inp.device):
        masked_select_kernel[grid](inp, mask_flattened, prefix_sum, out, n_elements)
    return out
