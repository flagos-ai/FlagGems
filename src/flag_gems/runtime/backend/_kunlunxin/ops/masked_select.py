import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def heur_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["n_elements"], 12))  # cluster_num


@libentry()
@triton.heuristics(
    values={
        "BLOCK_SIZE": heur_block_size,
    },
)
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

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(tl.int1)
    out_offset = (
        tl.load(prefix_sum_ptr + offsets, mask=select_mask and mask, other=0.0) - 1
    )

    tl.store(out_ptr + out_offset, inp, mask=(select_mask and mask))


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

    prefix_sum = mask_flattened.cumsum(axis=0)
    out = torch.empty(prefix_sum[-1].item(), dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    import os

    os.environ["TRITONXPU_OTHER_SIM"] = "1"
    os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
    with torch_device_fn.device(inp.device):
        masked_select_kernel[grid](inp, mask_flattened, prefix_sum, out, n_elements)

    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]
    if "TRITONXPU_STORE_MASK_SIM" in os.environ:
        del os.environ["TRITONXPU_STORE_MASK_SIM"]

    return out
