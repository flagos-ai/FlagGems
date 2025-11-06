import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
def masked_fill_kernel(
    inp,
    expand_mask,
    value,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # 计算需要处理的总块数
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB

    # 循环处理每个子块
    for sub_block_idx in range(num_sub_blocks):
        # 计算当前子块的偏移量
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        # 加载 input 和 mask
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # 先写入原始输入
        tl.store(out + offsets, input_vals, mask=mask)

        # 再在需要填充的位置覆盖写入 value
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(
            fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0)
        )
        tl.store(out + offsets, overwrite_vals, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
def masked_fill_kernel_self(
    inp, expand_mask, value, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr
):
    pid = tle.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # 计算需要处理的总块数
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB

    # 循环处理每个子块
    for sub_block_idx in range(num_sub_blocks):
        # 计算当前子块的偏移量
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        # 加载 expand_mask
        fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # 构造写入的值：fill_mask==1 用 value，fill_mask==0 保留原值
        orig = tl.load(inp + offsets, mask=mask, other=0)
        value_vec = tl.full([BLOCK_SIZE_SUB], value, dtype=orig.dtype)
        result = tl.where(fill_mask, value_vec, orig)

        # 存储结果
        tl.store(inp + offsets, result, mask=mask)


def masked_fill(inp, mask, value):
    logger.debug("GEMS_ASCEND MASKED FILL")
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
    logger.debug("GEMS_ASCEND MASKED FILL_")
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
