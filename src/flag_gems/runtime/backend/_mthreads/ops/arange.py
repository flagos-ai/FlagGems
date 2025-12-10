import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.arange import arange_start as default_arange_start
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

device_ = runtime.device
_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
}
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
]


@libentry()
@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["n_elements", "USE_INT64"])
@triton.jit(do_not_specialize=["start", "step"])
def arange_kernel(
    out_ptr,
    start,
    step,
    n_elements,
    IS_FLOAT: tl.constexpr,
    USE_INT64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_INT64:
        offsets = offsets.to(tl.int64)
        n_elements = tl.full((1,), n_elements, tl.int64)
    else:
        offsets = offsets.to(tl.int32)
        n_elements = tl.full((1,), n_elements, tl.int32)
    mask = offsets < n_elements

    if IS_FLOAT:
        idx = offsets.to(tl.float32)
        step_val = tl.full((1,), step, tl.float32)
        start_val = tl.full((1,), start, tl.float32)
        values = tl.fma(idx, step_val, start_val)
    else:
        value_dtype = tl.int64 if USE_INT64 else tl.int32
        idx = offsets.to(value_dtype)
        step_val = tl.full((1,), step, value_dtype)
        start_val = tl.full((1,), start, value_dtype)
        values = start_val + idx * step_val

    tl.store(out_ptr + offsets, values, mask=mask)


def _normalize_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def _compute_size(start, end, step, is_float_dtype: bool) -> int:
    if step == 0:
        raise ValueError("arange(): step must be non-zero.")
    if is_float_dtype:
        size = math.ceil((end - start) / step)
    else:
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    return int(size) if size > 0 else 0


def _use_triton(dtype: torch.dtype, device: torch.device, size: int) -> bool:
    if device.type != "musa":
        return False
    if dtype not in _SUPPORTED_DTYPES:
        return False
    return size > 0


def _launch_triton_kernel(
    out: torch.Tensor,
    start,
    step,
    size: int,
    *,
    is_float_dtype: bool,
    use_int64: bool,
):
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        arange_kernel[grid](
            out,
            start,
            step,
            size,
            IS_FLOAT=is_float_dtype,
            USE_INT64=use_int64,
        )
    return out


def arange_start(
    start,
    end,
    step=1,
    *,
    dtype: Optional[torch.dtype] = None,
    layout=None,
    device=None,
    pin_memory: Optional[bool] = None,
):
    logger.debug("GEMS_MTHREADS ARANGE")
    start = _normalize_scalar(start)
    end = _normalize_scalar(end)
    step = _normalize_scalar(step)

    if dtype is None:
        dtype = torch.int64
    if pin_memory is None:
        pin_memory = False
    if device is None:
        device = torch.device(device_.name)
    else:
        device = torch.device(device)

    is_float_dtype = torch.is_floating_point(torch.tensor(0, dtype=dtype))
    use_int64 = dtype == torch.int64
    size = _compute_size(start, end, step, is_float_dtype)

    if not _use_triton(dtype, device, size):
        return default_arange_start(
            start,
            end,
            step,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    return _launch_triton_kernel(
        result,
        start,
        step,
        size,
        is_float_dtype=is_float_dtype,
        use_int64=use_int64,
    )


def arange(
    end,
    *,
    dtype: Optional[torch.dtype] = None,
    layout=None,
    device=None,
    pin_memory: Optional[bool] = None,
):
    return arange_start(
        0,
        end,
        1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
