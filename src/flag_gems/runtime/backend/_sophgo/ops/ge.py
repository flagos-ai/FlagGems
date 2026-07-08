import logging

import triton
import triton.language as tl

from flag_gems.runtime import device
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)

_SOPHGO_GRID_CAP = 64
_SMALL_TILE = 4096
_LARGE_TILE = 8192
_DEVICE_NAME = device.name


def _config(max_tile_size):
    return CodeGenConfig(
        max_tile_size,
        (_SOPHGO_GRID_CAP, 1, 1),
        32,
        False,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    )


_small_config = _config(_SMALL_TILE)
_large_config = _config(_LARGE_TILE)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=_small_config)
@triton.jit
def _ge_func_small(x, y):
    return x.to(tl.float32) >= y


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=_large_config)
@triton.jit
def _ge_func_large(x, y):
    return x.to(tl.float32) >= y


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_small_config,
)
@triton.jit
def _ge_func_scalar_small(x, y):
    return x.to(tl.float32) >= y


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=_large_config,
)
@triton.jit
def _ge_func_scalar_large(x, y):
    return x.to(tl.float32) >= y


def _move_to_same_device(a, b):
    if a.device == b.device:
        return a, b
    if a.device.type == _DEVICE_NAME:
        return a, b.to(a.device)
    return a.to(b.device), b


def ge(A, B):
    logger.debug("SOPHGO GEMS GE")
    A, B = _move_to_same_device(A, B)
    if A.numel() <= _SMALL_TILE:
        return _ge_func_small(A, B)
    return _ge_func_large(A, B)


def ge_scalar(A, B):
    logger.debug("SOPHGO GEMS GE SCALAR")
    if A.numel() <= _SMALL_TILE:
        return _ge_func_scalar_small(A, B)
    return _ge_func_scalar_large(A, B)
