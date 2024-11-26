import logging

import torch
import triton
import triton.language as tl

from ..utils import broadcastable_to, libentry


def cfggen():
    block_size = [1024, 2048, 4096]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_SIZE": n}, num_warps=w) for n in block_size for w in warps
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["N"])
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
@triton.autotune(configs=cfggen(), key=["N"])
@triton.jit
def masked_fill_kernel_self(inp, expand_mask, value, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    tl.store(inp + offsets, value, fill_mask and mask)


def masked_fill(inp, mask, value):
    logging.debug("GEMS MASKED FILL")
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
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, N)
    return out


def masked_fill_(inp, mask, value):
    logging.debug("GEMS MASKED FILL")
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
