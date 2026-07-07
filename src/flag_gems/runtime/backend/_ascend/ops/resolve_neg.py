import logging
import math

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.jit
def resolve_neg_kernel(
    inp,
    out,
    data_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_DATA_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    iter_num = tl.cdiv(BLOCK_SIZE, MAX_DATA_SIZE)

    for idx in tl.range(0, iter_num):
        offsets = pid * BLOCK_SIZE + idx * MAX_DATA_SIZE + tl.arange(0, MAX_DATA_SIZE)
        mask = offsets < data_len
        inp_val = tl.load(inp + offsets, mask=mask)
        out_val = -inp_val
        tl.store(out + offsets, out_val, mask=mask)


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS_ASCEND RESOLVE_NEG")

    if A.is_neg():
        data_len = A.numel()
        out = torch.empty(A.numel(), dtype=A.dtype, device=A.device)

        CORE_NUM = get_npu_properties()["num_vectorcore"]
        BLOCK_SIZE = math.ceil(data_len / CORE_NUM)
        MAX_DATA_SIZE = 20 * 1024

        grid = lambda meta: (triton.cdiv(data_len, meta["BLOCK_SIZE"]),)
        resolve_neg_kernel[grid](
            A,
            out,
            data_len,
            BLOCK_SIZE,
            MAX_DATA_SIZE,
        )
        out = out.view(A.shape)
        return out
    else:
        return A
