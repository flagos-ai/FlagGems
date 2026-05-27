import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

device_ = device
logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

ALL_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
ALL_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)

# Threshold for switching between pointwise_dynamic (small tensors)
# and hand-written multi-core kernel (large tensors).
SMALL_TENSOR_THRESHOLD = 100000


def check_dtype(fill_value, dtype, device):
    if isinstance(fill_value, bool):
        if dtype != torch.bool:
            fill_value = int(fill_value)

    elif (
        dtype in ALL_INT_DTYPES
        and (fill_value < torch.iinfo(dtype).min or fill_value > torch.iinfo(dtype).max)
    ) or (
        dtype in ALL_FLOAT_DTYPES
        and not (math.isinf(fill_value) or math.isnan(fill_value))
        and (fill_value < torch.finfo(dtype).min or fill_value > torch.finfo(dtype).max)
    ):
        raise RuntimeError(
            f"value cannot be converted to type {dtype} without overflow"
        )

    return fill_value


# Small tensor path: pointwise_dynamic has lower launch overhead
@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def full_func(out, fill_value):
    return fill_value


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def full_func_scalar(out, fill_value):
    return tl.full(out.shape, fill_value, out.dtype)


# Large tensor path: hand-written multi-core kernel for better throughput
@libentry()
@triton.jit(do_not_specialize=["fill_value"])
def full_kernel(
    out_ptr,
    N,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
    SUBBLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    pid_offset = pid * BLOCK_SIZE
    cols = tl.arange(0, SUBBLOCK_SIZE)
    num_loop = triton.cdiv(BLOCK_SIZE, SUBBLOCK_SIZE)
    for iloop in tl.range(num_loop):
        offset = pid_offset + iloop * SUBBLOCK_SIZE + cols
        tl.store(out_ptr + offset, fill_value, mask=offset < N)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ASCEND FULL")
    if device is None:
        device = torch.device(device_.name)
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = torch.bool
        elif isinstance(fill_value, int):
            dtype = torch.int64
        else:
            dtype = torch.get_default_dtype()
    else:
        fill_value = check_dtype(fill_value, dtype, device)

    out = torch.empty(size, device=device, dtype=dtype)
    N = out.numel()
    if N == 0:
        return out

    if N < SMALL_TENSOR_THRESHOLD:
        # Small tensor: use pointwise_dynamic for lower launch overhead
        if isinstance(fill_value, torch.Tensor):
            return full_func(out, fill_value, out0=out)
        else:
            return full_func_scalar(out, fill_value, out0=out)

    # Large tensor: use hand-written multi-core kernel
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()

    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(device):
        full_kernel[grid,](out, N, fill_value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return out
