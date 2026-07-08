import logging
from numbers import Integral

import torch
import triton
import triton.language as tl

from flag_gems.ops.bitwise_or import (
    bitwise_or_scalar as _fallback_bitwise_or_scalar,
    bitwise_or_scalar_ as _fallback_bitwise_or_scalar_,
    bitwise_or_scalar_tensor as _fallback_bitwise_or_scalar_tensor,
    bitwise_or_tensor as _fallback_bitwise_or_tensor,
    bitwise_or_tensor_ as _fallback_bitwise_or_tensor_,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SOPHGO_GRID_CAP = 64
_SMALL_BLOCK_SIZE = 4096
_LARGE_BLOCK_SIZE = 8192
_INTEGER_FAST_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@libentry()
@triton.jit
def _bitwise_or_tensor_contig_kernel(
    a,
    b,
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
        x = tl.load(a + offsets, mask=mask)
        y = tl.load(b + offsets, mask=mask)
        tl.store(out + offsets, x | y, mask=mask)


@libentry()
@triton.jit
def _bitwise_or_tensor_contig_nomask_kernel(
    a,
    b,
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
        x = tl.load(a + offsets)
        y = tl.load(b + offsets)
        tl.store(out + offsets, x | y)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _bitwise_or_scalar_contig_kernel(
    a,
    value,
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
        x = tl.load(a + offsets, mask=mask)
        tl.store(out + offsets, x | value, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["value"])
def _bitwise_or_scalar_contig_nomask_kernel(
    a,
    value,
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
        x = tl.load(a + offsets)
        tl.store(out + offsets, x | value)


def _is_tpu_tensor(x):
    return isinstance(x, torch.Tensor) and x.device.type in ("tpu", "sophgo")


def _is_integer_fast_dtype(dtype):
    return dtype in _INTEGER_FAST_DTYPES


def _can_use_tensor_fast_path(a, b, out=None):
    if not (_is_tpu_tensor(a) and _is_tpu_tensor(b)):
        return False
    if a.device != b.device or a.dtype != b.dtype:
        return False
    if not _is_integer_fast_dtype(a.dtype):
        return False
    if a.shape != b.shape or not (a.is_contiguous() and b.is_contiguous()):
        return False
    if out is not None:
        return (
            _is_tpu_tensor(out)
            and out.device == a.device
            and out.shape == a.shape
            and out.dtype == a.dtype
            and out.is_contiguous()
        )
    return True


def _can_use_scalar_fast_path(a, value, out=None):
    if not (_is_tpu_tensor(a) and isinstance(value, Integral)):
        return False
    if not (_is_integer_fast_dtype(a.dtype) and a.is_contiguous()):
        return False
    if out is not None:
        return (
            _is_tpu_tensor(out)
            and out.device == a.device
            and out.shape == a.shape
            and out.dtype == a.dtype
            and out.is_contiguous()
        )
    return torch.result_type(a, value) == a.dtype


def _choose_block_size(n_elements):
    return _SMALL_BLOCK_SIZE if n_elements <= _SMALL_BLOCK_SIZE else _LARGE_BLOCK_SIZE


def _launch_grid(n_elements, block_size):
    return (min(triton.cdiv(n_elements, block_size), _SOPHGO_GRID_CAP),)


def _launch_tensor(a, b, out):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block_size = _choose_block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _bitwise_or_tensor_contig_nomask_kernel[grid](
            a, b, out, n_elements, BLOCK_SIZE=block_size
        )
    else:
        _bitwise_or_tensor_contig_kernel[grid](
            a, b, out, n_elements, BLOCK_SIZE=block_size
        )
    return out


def _launch_scalar(a, value, out):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block_size = _choose_block_size(n_elements)
    grid = _launch_grid(n_elements, block_size)
    if n_elements % block_size == 0:
        _bitwise_or_scalar_contig_nomask_kernel[grid](
            a, int(value), out, n_elements, BLOCK_SIZE=block_size
        )
    else:
        _bitwise_or_scalar_contig_kernel[grid](
            a, int(value), out, n_elements, BLOCK_SIZE=block_size
        )
    return out


def bitwise_or_tensor(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR")
    if _can_use_tensor_fast_path(A, B):
        return _launch_tensor(A, B, torch.empty_like(A))
    return _fallback_bitwise_or_tensor(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR_")
    if _can_use_tensor_fast_path(A, B, A):
        return _launch_tensor(A, B, A)
    return _fallback_bitwise_or_tensor_(A, B)


def bitwise_or_scalar(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR SCALAR")
    if _can_use_scalar_fast_path(A, B):
        return _launch_scalar(A, B, torch.empty_like(A))
    return _fallback_bitwise_or_scalar(A, B)


def bitwise_or_scalar_(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR_ SCALAR")
    if _can_use_scalar_fast_path(A, B, A):
        return _launch_scalar(A, B, A)
    return _fallback_bitwise_or_scalar_(A, B)


def bitwise_or_scalar_tensor(A, B):
    logger.debug("SOPHGO GEMS BITWISE OR SCALAR TENSOR")
    if _can_use_scalar_fast_path(B, A):
        return _launch_scalar(B, A, torch.empty_like(B))
    return _fallback_bitwise_or_scalar_tensor(A, B)
