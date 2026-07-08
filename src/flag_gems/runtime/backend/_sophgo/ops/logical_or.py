import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.logical_or import logical_or as _fallback_logical_or
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)

# sg2260 has a physical one-dimensional grid limit of 64.  Keeping exactly
# this many programs resident and letting each program process several tiles
# avoids the launch/scheduling overhead of a larger logical grid.
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
    """Use large 1-D tiles for pointwise fallbacks as well."""
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
def _logical_or_contig_masked_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    total_tiles,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BOOL_INPUTS: tl.constexpr,
):
    pid = tle.program_id(0)

    # A fixed physical grid combined with this loop covers arbitrarily large
    # tensors without creating waves beyond sg2260's 64-program grid.
    for tile_id in range(pid, total_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask, other=0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0)
        if BOOL_INPUTS:
            out = a.to(tl.int1).logical_or(b.to(tl.int1))
        else:
            # Logical operators use non-zero truthiness, including NaN.
            out = (a != 0).logical_or(b != 0)
        tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def _logical_or_contig_nomask_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    total_tiles,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BOOL_INPUTS: tl.constexpr,
):
    pid = tle.program_id(0)

    for tile_id in range(pid, total_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(a_ptr + offsets)
        b = tl.load(b_ptr + offsets)
        if BOOL_INPUTS:
            out = a.to(tl.int1).logical_or(b.to(tl.int1))
        else:
            out = (a != 0).logical_or(b != 0)
        tl.store(out_ptr + offsets, out)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=_small_config
)
@triton.jit
def _logical_or_func_small(x, y):
    return x.to(tl.int1).logical_or(y.to(tl.int1))


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=_large_config
)
@triton.jit
def _logical_or_func_large(x, y):
    return x.to(tl.int1).logical_or(y.to(tl.int1))


def _is_tpu_tensor(x):
    return isinstance(x, torch.Tensor) and x.device.type in ("tpu", "sophgo")


def _move_to_same_device(a, b):
    if a.device == b.device:
        return a, b
    if a.device.type == _DEVICE_NAME:
        return a, b.to(a.device)
    return a.to(b.device), b


def _can_use_fast_path(a, b):
    return (
        _is_tpu_tensor(a)
        and _is_tpu_tensor(b)
        and a.device == b.device
        and a.dtype in _FAST_DTYPES
        and b.dtype in _FAST_DTYPES
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
    )


def _choose_block_size(n_elements):
    # 4K is sufficient for short tensors; 8K amortizes per-tile overhead for
    # the inference tensors that dominate this operator's use.
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
    kernel = (
        _logical_or_contig_nomask_kernel
        if n_elements % block_size == 0
        else _logical_or_contig_masked_kernel
    )
    with torch_device_fn.device(a.device):
        kernel[(grid_size,)](
            a,
            b,
            out,
            n_elements,
            total_tiles,
            BLOCK_SIZE=block_size,
            GRID_SIZE=grid_size,
            BOOL_INPUTS=(a.dtype == torch.bool and b.dtype == torch.bool),
        )
    return out


def _broadcast_numel(a, b):
    return math.prod(torch.broadcast_shapes(a.shape, b.shape))


def logical_or(A, B):
    """Sophgo implementation of ``torch.logical_or``.

    The dedicated path intentionally only handles layout-preserving cases.
    PointwiseDynamic remains responsible for broadcasting, views, and dtypes
    whose direct Triton lowering has not been qualified on sg2260.
    """
    logger.debug("SOPHGO GEMS LOGICAL_OR")
    A, B = _move_to_same_device(A, B)
    if _can_use_fast_path(A, B):
        return _launch_fast_path(A, B)

    if _broadcast_numel(A, B) <= _SMALL_BLOCK_SIZE:
        return _logical_or_func_small(A, B)
    return _logical_or_func_large(A, B)
