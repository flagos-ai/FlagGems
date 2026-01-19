import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def masked_fill_kernel_heur_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))  # cluster_num


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
# @triton.heuristics(
#     values={
#         "BLOCK_SIZE": masked_fill_kernel_heur_block_size,
#     },
# )
@triton.jit
def masked_fill_kernel(
    inp, expand_mask, value, out, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(not fill_mask) and mask, other=0)
    out_offset_1 = tl.where((not fill_mask) and mask, offsets, -1)
    tl.store(out + out_offset_1, cur_inp, (not fill_mask) and mask)
    out_offset_2 = tl.where(fill_mask and mask, offsets, -1)
    tl.store(out + out_offset_2, value, fill_mask and mask)


def masked_fill_kernel_self_heur_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))  # cluster_num


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
# @triton.heuristics(
#     values={
#         "BLOCK_SIZE": masked_fill_kernel_self_heur_block_size,
#     },
# )
@triton.jit
def masked_fill_kernel_self(inp, expand_mask, value, N, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    tl.store(inp + offsets, value, fill_mask and mask)


def masked_fill(inp, mask, value):
    logger.debug("GEMS MASKED FILL")
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
    grid = 12
    BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(N, grid))

    import os

    os.environ["TRITONXPU_OTHER_SIM"] = "1"
    os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
    masked_fill_kernel[grid,](
        inp,
        expand_mask.to(torch.int),
        value,
        out,
        N,
        BLOCK_SIZE,
        isCloseUnrollControl=True,
        buffer_size_limit=2048,
    )

    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]
    if "TRITONXPU_STORE_MASK_SIM" in os.environ:
        del os.environ["TRITONXPU_STORE_MASK_SIM"]
    return out


def masked_fill_(inp, mask, value):
    logger.debug("GEMS MASKED FILL")
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

    import os

    os.environ["TRITONXPU_OTHER_SIM"] = "1"
    os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"

    grid = 12
    BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(N, grid))
    masked_fill_kernel_self[grid,](
        inp, expand_mask.to(torch.int), value, N, BLOCK_SIZE, buffer_size_limit=2048
    )
    if "TRITONXPU_OTHER_SIM" in os.environ:
        del os.environ["TRITONXPU_OTHER_SIM"]
    if "TRITONXPU_STORE_MASK_SIM" in os.environ:
        del os.environ["TRITONXPU_STORE_MASK_SIM"]
    return inp
