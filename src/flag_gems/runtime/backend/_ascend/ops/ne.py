import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

CORE_NUM = 40

try:
    import torch_npu  # noqa: F401
    import triton.runtime.driver as driver

    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    CORE_NUM = props["num_vectorcore"]
except Exception:
    CORE_NUM = 40


@triton.jit
def ne_scalar_kernel(
    inp,
    out,
    scalar_val,
    data_len,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    iter_num = tl.cdiv(BLOCK_SIZE, TILE_SIZE)

    for idx in tl.range(0, iter_num):
        offsets = pid * BLOCK_SIZE + idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < data_len
        inp_val = tl.load(inp + offsets, mask=mask)
        result = inp_val.to(tl.float32) != scalar_val
        tl.store(out + offsets, result, mask=mask)


def ne_scalar(A, B):
    logger.debug("GEMS_ASCEND NE SCALAR")
    data_len = A.numel()
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)

    BLOCK_SIZE = math.ceil(data_len / CORE_NUM)
    TILE_SIZE = 8192
    BLOCK_SIZE = max(BLOCK_SIZE, TILE_SIZE)

    grid = lambda meta: (triton.cdiv(data_len, meta["BLOCK_SIZE"]),)
    ne_scalar_kernel[grid](
        A.contiguous().view(-1),
        out.view(-1),
        float(B),
        data_len,
        BLOCK_SIZE,
        TILE_SIZE,
    )
    return out


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logger.debug("GEMS_ASCEND NE")
    return ne_func(A, B)
