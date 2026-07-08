import logging
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems.ops.fill import (
    fill_scalar as _fallback_fill_scalar,
    fill_scalar_ as _fallback_fill_scalar_,
    fill_tensor as _fallback_fill_tensor,
    fill_tensor_ as _fallback_fill_tensor_,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SOPHGO_GRID_CAP = 64


@libentry()
@triton.jit(do_not_specialize=["value"])
def _fill_scalar_contig_kernel(
    out,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(out + offsets, value, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _fill_scalar_contig_nomask_kernel(
    out,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        tl.store(out + offsets, value)


@libentry()
@triton.jit
def _fill_tensor_contig_kernel(
    out,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    value_scalar = tl.load(value)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(out + offsets, value_scalar, mask=mask)


@libentry()
@triton.jit
def _fill_tensor_contig_nomask_kernel(
    out,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    value_scalar = tl.load(value)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        tl.store(out + offsets, value_scalar)


def _is_tpu_tensor(x):
    return isinstance(x, torch.Tensor) and x.device.type in ("tpu", "sophgo")


def _can_use_contiguous_float32_tensor(x):
    return _is_tpu_tensor(x) and x.dtype is torch.float32 and x.is_contiguous()


def _choose_block_size(n_elements):
    return 4096 if n_elements <= 4096 else 8192


def _launch_grid(n_elements, block_size):
    return (min(triton.cdiv(n_elements, block_size), _SOPHGO_GRID_CAP),)


def _launch_scalar(out, value):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block_size = _choose_block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _fill_scalar_contig_nomask_kernel[grid](
            out, n_elements, float(value), BLOCK_SIZE=block_size
        )
    else:
        _fill_scalar_contig_kernel[grid](
            out, n_elements, float(value), BLOCK_SIZE=block_size
        )
    return out


def _launch_tensor(out, value):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block_size = _choose_block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _fill_tensor_contig_nomask_kernel[grid](
            out, n_elements, value, BLOCK_SIZE=block_size
        )
    else:
        _fill_tensor_contig_kernel[grid](
            out, n_elements, value, BLOCK_SIZE=block_size
        )
    return out


def fill_scalar(input, value):
    logger.debug("SOPHGO FILL_SCALAR")
    if _can_use_contiguous_float32_tensor(input) and isinstance(value, Number):
        return _launch_scalar(torch.empty_like(input), value)
    return _fallback_fill_scalar(input, value)


def fill_scalar_(self, value):
    logger.debug("SOPHGO FILL_SCALAR_")
    if _can_use_contiguous_float32_tensor(self) and isinstance(value, Number):
        return _launch_scalar(self, value)
    return _fallback_fill_scalar_(self, value)


def fill_tensor(input, value):
    logger.debug("SOPHGO FILL_TENSOR")
    if not isinstance(value, torch.Tensor):
        return fill_scalar(input, value)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if _can_use_contiguous_float32_tensor(input):
        out = torch.empty_like(input)
        if _is_tpu_tensor(value):
            return _launch_tensor(out, value)
        return _launch_scalar(out, value.item())
    return _fallback_fill_tensor(input, value)


def fill_tensor_(self, value):
    logger.debug("SOPHGO FILL_TENSOR_")
    if not isinstance(value, torch.Tensor):
        return fill_scalar_(self, value)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if _can_use_contiguous_float32_tensor(self):
        if _is_tpu_tensor(value):
            return _launch_tensor(self, value)
        return _launch_scalar(self, value.item())
    return _fallback_fill_tensor_(self, value)
