import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.logical_and import logical_and as _fallback_logical_and
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)

_SOPHGO_GRID_CAP = 64
_SMALL_BLOCK_SIZE = 4096
_LARGE_BLOCK_SIZE = 8192
_DEVICE_NAME = device.name
_FAST_DTYPES = {
    torch.bool,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
}


def _config(max_tile_size):
    return CodeGenConfig(
        max_tile_size,
        (_SOPHGO_GRID_CAP, 1, 1),
        32,
        False,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    )


_small_config = _config(_SMALL_BLOCK_SIZE)
_large_config = _config(_LARGE_BLOCK_SIZE)


@libentry()
@triton.jit
def _logical_and_contig_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    total_tiles,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    BOOL_INPUTS: tl.constexpr,
):
    pid = tle.program_id(0)

    for tile_id in range(pid, total_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if DIVISIBLE_N:
            a = tl.load(a_ptr + offsets)
            b = tl.load(b_ptr + offsets)
            if BOOL_INPUTS:
                out = a.to(tl.int1).logical_and(b.to(tl.int1))
            else:
                out = (a != 0).logical_and(b != 0)
            tl.store(out_ptr + offsets, out)
        else:
            mask = offsets < n_elements
            a = tl.load(a_ptr + offsets, mask=mask, other=0)
            b = tl.load(b_ptr + offsets, mask=mask, other=0)
            if BOOL_INPUTS:
                out = a.to(tl.int1).logical_and(b.to(tl.int1))
            else:
                out = (a != 0).logical_and(b != 0)
            tl.store(out_ptr + offsets, out, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=_small_config)
@triton.jit
def _logical_and_func_small(x, y):
    return x.to(tl.int1).logical_and(y.to(tl.int1))


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=_large_config)
@triton.jit
def _logical_and_func_large(x, y):
    return x.to(tl.int1).logical_and(y.to(tl.int1))


def _is_tpu_tensor(x):
    return isinstance(x, torch.Tensor) and x.device.type in ("tpu", "sophgo")


def _move_to_same_device(a, b):
    if a.device == b.device:
        return a, b
    if a.device.type == _DEVICE_NAME:
        return a, b.to(a.device)
    return a.to(b.device), b


def _supported_fast_dtype(dtype):
    return dtype in _FAST_DTYPES


def _can_use_fast_path(a, b):
    return (
        _is_tpu_tensor(a)
        and _is_tpu_tensor(b)
        and a.device == b.device
        and _supported_fast_dtype(a.dtype)
        and _supported_fast_dtype(b.dtype)
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
    )


def _choose_block_size(n_elements):
    return _SMALL_BLOCK_SIZE if n_elements <= _SMALL_BLOCK_SIZE else _LARGE_BLOCK_SIZE


def _launch_grid(n_elements, block_size):
    return min(triton.cdiv(n_elements, block_size), _SOPHGO_GRID_CAP)


def _launch_fast_path(a, b):
    out = torch.empty_like(a, dtype=torch.bool)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    block_size = _choose_block_size(n_elements)
    total_tiles = triton.cdiv(n_elements, block_size)
    grid_size = _launch_grid(n_elements, block_size)
    with torch_device_fn.device(a.device):
        _logical_and_contig_kernel[(grid_size,)](
            a,
            b,
            out,
            n_elements,
            total_tiles,
            BLOCK_SIZE=block_size,
            GRID_SIZE=grid_size,
            DIVISIBLE_N=(n_elements % block_size == 0),
            BOOL_INPUTS=(a.dtype == torch.bool and b.dtype == torch.bool),
        )
    return out


def logical_and(A, B):
    logger.debug("SOPHGO GEMS LOGICAL_AND")
    A, B = _move_to_same_device(A, B)
    if A.numel() <= _SMALL_BLOCK_SIZE:
        return _fallback_logical_and(A, B)
    if _can_use_fast_path(A, B):
        return _launch_fast_path(A, B)
    return _logical_and_func_large(A, B)
