import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


ALL_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
ALL_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


@libentry()
@triton.jit(do_not_specialize=["fill_value_or_ptr"])
def full_kernel(
    output_ptr,
    n_elements,
    fill_value_or_ptr,
    FILL_VALUE_IS_PTR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if FILL_VALUE_IS_PTR:
        fill_value = tl.load(fill_value_or_ptr)
    else:
        fill_value = fill_value_or_ptr
    tl.store(output_ptr + offsets, fill_value, mask=mask)


def check_dtype(fill_value, dtype, device):
    if isinstance(fill_value, bool):
        if dtype != torch.bool:
            fill_value = int(fill_value)
    elif (
        dtype in ALL_INT_DTYPES
        and (fill_value < torch.iinfo(dtype).min or fill_value > torch.iinfo(dtype).max)
    ) or (
        dtype in ALL_FLOAT_DTYPES
        and (fill_value < torch.finfo(dtype).min or fill_value > torch.finfo(dtype).max)
    ):
        raise RuntimeError(
            f"value cannot be converted to type {dtype} without overflow"
        )
    if dtype == torch.float64:
        fill_value = torch.tensor(fill_value, dtype=dtype, device=device)
    return fill_value


def _launch_fill(out, fill_value):
    n_elements = out.numel()
    if n_elements == 0:
        return out

    fill_value_is_ptr = isinstance(fill_value, torch.Tensor)
    if fill_value_is_ptr and fill_value.numel() == 1 and out.dtype != torch.float64:
        fill_value = fill_value.item()
        fill_value_is_ptr = False

    block_size = 4096 if not fill_value_is_ptr else 1024
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(out.device):
        full_kernel[grid](
            out,
            n_elements,
            fill_value,
            FILL_VALUE_IS_PTR=fill_value_is_ptr,
            BLOCK_SIZE=block_size,
        )
    return out


def full_like(
    x,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    logging.debug("GEMS FULL_LIKE SOPHGO")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty_like(x, device=device, dtype=dtype)
    return _launch_fill(out, fill_value)
