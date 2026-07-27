# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _range_kernel(out_ptr, start, size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offsets, offsets + start, mask=offsets < size)


def range(start, end, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_KUNLUNXIN RANGE")
    if layout not in (None, torch.strided):
        raise RuntimeError("torch.range only supports strided layout")
    if pin_memory:
        raise RuntimeError("torch.range does not support pinned memory on XPU")
    if dtype is None:
        dtype = torch.float64 if any(isinstance(value, float) for value in (start, end)) else torch.int64
    if dtype == torch.bfloat16:
        raise RuntimeError("torch.range does not support bfloat16 on XPU")
    if dtype not in (
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError(f"torch.range does not support dtype {dtype}")

    integer_output = dtype in (torch.int32, torch.int64)
    start_value = int(start) if integer_output else float(start)
    end_value = int(end) if integer_output else float(end)
    size = end_value - start_value + 1 if integer_output else math.ceil(end_value - start_value) + 1
    size = max(0, int(size))
    output_device = device if device is not None else runtime.device.name
    out = torch.empty(size, dtype=dtype, device=output_device)
    if size == 0:
        return out
    with torch_device_fn.device(out.device):
        _range_kernel[(triton.cdiv(size, 256),)](out, start_value, size, BLOCK_SIZE=256)
    return out
