import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import broadcastable_to, libentry, libtuner

from ..utils import MAX_GRID_SIZE_X

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("masked_fill"),
    key=["N"],
    strategy=["log"],
)
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(not fill_mask) and mask, other=0)
    tl.store(out + offsets, cur_inp, (not fill_mask) and mask)
    tl.store(out + offsets, value, fill_mask and mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("masked_fill"),
    key=["N"],
    strategy=["log"],
)
@triton.jit
def masked_fill_kernel_self(inp, expand_mask, value, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=1) * tl.num_programs(0) + tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_val = tl.full((BLOCK_SIZE,), value, dtype=inp.dtype.element_ty)
    tl.store(inp + offsets, cur_val, fill_mask and mask)


def masked_fill(inp, mask, value):
    logger.debug("GEMS_CAMBRICON MASKED FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )

    inp = inp.contiguous()
    mask = mask.contiguous()
    expand_mask = mask.expand(inp.shape)
    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)

    N = inp.numel()
    if N == 0:
        return out

    def gridfn(meta):
        blocks = triton.cdiv(N, meta["BLOCK_SIZE"])
        x = min(MAX_GRID_SIZE_X, blocks)
        y = triton.cdiv(blocks, x)
        return (x, y, 1)

    masked_fill_kernel[gridfn](inp, expand_mask.to(torch.int), value, out, N)
    return out


def masked_fill_(inp, mask, value):
    logger.debug("GEMS_CAMBRICON MASKED FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        if mask.item():
            inp[()] = value
        return inp

    inp = inp.contiguous()
    mask = mask.contiguous()
    expand_mask = mask.expand(inp.shape)

    N = inp.numel()
    if N == 0:
        return inp
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    masked_fill_kernel_self[grid](inp, expand_mask.to(torch.int), value, N)
    return inp
