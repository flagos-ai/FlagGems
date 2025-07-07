import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


ALL_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
ALL_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


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


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def full_func(out, fill_value):
    return fill_value


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def full_func_scalar(out, fill_value):
    return tl.full(out.shape, fill_value, out.dtype)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS FULL")
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

    if isinstance(fill_value, torch.Tensor):
        return full_func(out, fill_value, out0=out)
    else:
        return full_func_scalar(out, fill_value, out0=out)
