import logging
import math
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems.ops.div import (
    div_mode as _fallback_div_mode,
    div_mode_ as _fallback_div_mode_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    true_divide as _fallback_true_divide,
    true_divide_ as _fallback_true_divide_,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SOPHGO_GRID_CAP = 64


@libentry()
@triton.jit
def _div_tt_contig_kernel(
    x,
    y,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_val = tl.load(x + offsets, mask=mask, other=0.0)
        y_val = tl.load(y + offsets, mask=mask, other=1.0)
        tl.store(out + offsets, x_val / y_val, mask=mask)


@libentry()
@triton.jit
def _div_tt_contig_nomask_kernel(
    x,
    y,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        x_val = tl.load(x + offsets)
        y_val = tl.load(y + offsets)
        tl.store(out + offsets, x_val / y_val)


@libentry()
@triton.jit
def _div_ts_contig_kernel(
    x,
    inv_y_scalar,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_val = tl.load(x + offsets, mask=mask, other=0.0)
        tl.store(out + offsets, x_val * inv_y_scalar, mask=mask)


@libentry()
@triton.jit
def _div_ts_contig_nomask_kernel(
    x,
    inv_y_scalar,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        x_val = tl.load(x + offsets)
        tl.store(out + offsets, x_val * inv_y_scalar)


@libentry()
@triton.jit
def _div_st_contig_kernel(
    x_scalar,
    y,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        y_val = tl.load(y + offsets, mask=mask, other=1.0)
        tl.store(out + offsets, x_scalar / y_val, mask=mask)


@libentry()
@triton.jit
def _div_st_contig_nomask_kernel(
    x_scalar,
    y,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_jobs = tle.num_programs(0)
    block_start = (pid * BLOCK_SIZE).to(tl.int64)
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        y_val = tl.load(y + offsets)
        tl.store(out + offsets, x_scalar / y_val)


def _is_tpu_tensor(x):
    return isinstance(x, torch.Tensor) and x.device.type in ("tpu", "sophgo")


def _can_use_contiguous_float32_tensor(x):
    return _is_tpu_tensor(x) and x.dtype is torch.float32 and x.is_contiguous()


def _block_size(n_elements):
    return 4096 if n_elements <= 4096 else 8192


def _launch_grid(n_elements, block_size):
    return (min(triton.cdiv(n_elements, block_size), _SOPHGO_GRID_CAP),)


def _launch_tt(A, B, out):
    n_elements = A.numel()
    if n_elements == 0:
        return out
    block_size = _block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _div_tt_contig_nomask_kernel[grid](
            A, B, out, n_elements, BLOCK_SIZE=block_size
        )
    else:
        _div_tt_contig_kernel[grid](A, B, out, n_elements, BLOCK_SIZE=block_size)
    return out


def _launch_ts(A, B, out):
    n_elements = A.numel()
    if n_elements == 0:
        return out
    block_size = _block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    b = float(B)
    inv_b = math.copysign(float("inf"), b) if b == 0.0 else 1.0 / b
    if n_elements % block_size == 0:
        _div_ts_contig_nomask_kernel[grid](
            A, inv_b, out, n_elements, BLOCK_SIZE=block_size
        )
    else:
        _div_ts_contig_kernel[grid](
            A, inv_b, out, n_elements, BLOCK_SIZE=block_size
        )
    return out


def _launch_st(A, B, out):
    n_elements = B.numel()
    if n_elements == 0:
        return out
    block_size = _block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _div_st_contig_nomask_kernel[grid](
            float(A), B, out, n_elements, BLOCK_SIZE=block_size
        )
    else:
        _div_st_contig_kernel[grid](
            float(A), B, out, n_elements, BLOCK_SIZE=block_size
        )
    return out


def true_divide(A, B):
    logger.debug("SOPHGO TRUE_DIVIDE")
    if (
        _can_use_contiguous_float32_tensor(A)
        and _can_use_contiguous_float32_tensor(B)
        and A.shape == B.shape
    ):
        if A.numel() <= 4096:
            return _fallback_true_divide(A, B)
        return _launch_tt(A, B, torch.empty_like(A))
    if _can_use_contiguous_float32_tensor(A) and isinstance(B, Number):
        if A.numel() <= 4096:
            return _fallback_true_divide(A, B)
        return _launch_ts(A, B, torch.empty_like(A))
    if isinstance(A, Number) and _can_use_contiguous_float32_tensor(B):
        if B.numel() <= 4096:
            return _fallback_true_divide(A, B)
        return _launch_st(A, B, torch.empty_like(B))
    return _fallback_true_divide(A, B)


def true_divide_(A, B):
    logger.debug("SOPHGO TRUE_DIVIDE_")
    if (
        _can_use_contiguous_float32_tensor(A)
        and _can_use_contiguous_float32_tensor(B)
        and A.shape == B.shape
    ):
        if A.numel() <= 4096:
            return _fallback_true_divide_(A, B)
        return _launch_tt(A, B, A)
    if _can_use_contiguous_float32_tensor(A) and isinstance(B, Number):
        if A.numel() <= 4096:
            return _fallback_true_divide_(A, B)
        return _launch_ts(A, B, A)
    return _fallback_true_divide_(A, B)


def div_mode(A, B, rounding_mode=None):
    logger.debug("SOPHGO DIV_MODE")
    if rounding_mode is None:
        return true_divide(A, B)
    return _fallback_div_mode(A, B, rounding_mode)


def div_mode_(A, B, rounding_mode=None):
    logger.debug("SOPHGO DIV_MODE_")
    if rounding_mode is None:
        return true_divide_(A, B)
    return _fallback_div_mode_(A, B, rounding_mode)
