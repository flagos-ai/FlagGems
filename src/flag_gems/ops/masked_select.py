import logging

import torch
import triton
import triton.language as tl

from ..utils import broadcastable, libentry


def cfggen():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for w in [4, 8, 16, 32]
        for bs in [256, 512, 1024, 2048, 4096]
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["n_elements"])
@triton.jit
def masked_select_kernel(
    inp_ptr,
    select_mask_ptr,
    prefix_sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(tl.int1)
    out_offset = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0.0) - 1

    tl.store(out_ptr + out_offset, inp, mask=(select_mask and mask))


def masked_select(inp, mask):
    logging.debug("GEMS MASKED SELECT")

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
    with torch.cuda.device(inp.device):
        masked_select_kernel[grid](inp, mask_flattened, prefix_sum, out, n_elements)
    return out
