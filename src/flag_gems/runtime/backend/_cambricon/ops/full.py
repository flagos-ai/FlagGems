import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.shape_utils import volume

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 65536}, num_stages=1, num_warps=1),
    ],
    key=["n_elements"],
)
@triton.jit(do_not_specialize=["fill_value_or_ptr"])
def full_kernel(
    output_ptr,
    n_elements,
    fill_value_or_ptr,
    FILL_VALUE_IS_PTR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        if FILL_VALUE_IS_PTR:
            fill_value = tl.load(fill_value_or_ptr)
        else:
            fill_value = fill_value_or_ptr
        tl.store(output_ptr + offsets, fill_value, mask=mask)


ALL_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
ALL_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def check_dtype(fill_value, dtype, device):
    if isinstance(fill_value, bool):
        if dtype != torch.bool:
            fill_value = int(fill_value)
    else:
        if isinstance(fill_value, float) and math.isinf(fill_value):
            if dtype not in ALL_FLOAT_DTYPES:
                raise RuntimeError(
                    f"value {fill_value!r} cannot be converted to type {dtype} without overflow"
                )
        elif (
            dtype in ALL_INT_DTYPES
            and (
                fill_value < torch.iinfo(dtype).min
                or fill_value > torch.iinfo(dtype).max
            )
        ) or (
            dtype in ALL_FLOAT_DTYPES
            and (
                fill_value < torch.finfo(dtype).min
                or fill_value > torch.finfo(dtype).max
            )
        ):
            raise RuntimeError(
                f"value cannot be converted to type {dtype} without overflow"
            )
    if dtype in ALL_FLOAT_DTYPES:
        fill_value = torch.tensor(fill_value, dtype=dtype, device=device)
    return fill_value


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_CAMBRICON FULL")
    if device is None:
        device = torch.device("cpu")
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
    N = volume(size)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(device):
        full_kernel[grid_fn](
            out,
            N,
            fill_value,
            FILL_VALUE_IS_PTR=isinstance(fill_value, torch.Tensor),
        )
    return out
