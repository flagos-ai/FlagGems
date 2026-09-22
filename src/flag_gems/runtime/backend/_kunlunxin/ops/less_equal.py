import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.less_equal_ import less_equal_ as _generic_less_equal_
from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


config_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def less_equal_func(x, y):
    return x.to(tl.float32) <= y


def less_equal(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = less_equal_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def less_equal_func_scalar(x, y):
    return x.to(tl.float32) <= y


def less_equal_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        s = float(B)
        if (
            numel >= _LESS_EQUAL_SCALAR_FAST_TILE
            and numel % _LESS_EQUAL_SCALAR_FAST_TILE == 0
        ):
            return _less_equal_scalar_fast(
                A, s, (numel // _LESS_EQUAL_SCALAR_FAST_TILE,)
            )
        if (
            numel >= _LESS_EQUAL_SCALAR_MASKED_MIN
            and numel % _LESS_EQUAL_SCALAR_FAST_TILE != 0
        ):
            return _less_equal_scalar_fast_masked(A, s, numel)
    res = less_equal_func_scalar(A, B)
    return res


_LESS_EQUAL_SCALAR_FAST_TILE = 131072
_LESS_EQUAL_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def less_equal_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e38
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, 1.0 - t)


def _less_equal_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    less_equal_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_LESS_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def less_equal_scalar_fast_masked_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (x - scalar) * 1.0e38
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, 1.0 - t, mask=mask)


def _less_equal_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _LESS_EQUAL_SCALAR_FAST_TILE),)
    less_equal_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_LESS_EQUAL_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def less_equal_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = 1.0 - t
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def less_equal_(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_ TENSOR")
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel
            >= _LESS_EQUAL_TENSOR_INPLACE_FAST_TILE
            * _LESS_EQUAL_TENSOR_INPLACE_MIN_GRID
            and numel % _LESS_EQUAL_TENSOR_INPLACE_FAST_TILE == 0
        ):
            return _less_equal_tensor_inplace_fast(A, B, numel)
        less_equal_func_tensor_inplace(A, B, out0=A)
        return A
    return _generic_less_equal_(A, B)


_LESS_EQUAL_TENSOR_INPLACE_FAST_TILE = 131072
_LESS_EQUAL_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def less_equal_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = 1.0 - t
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _less_equal_tensor_inplace_fast(A, B, numel):
    grid = (numel // _LESS_EQUAL_TENSOR_INPLACE_FAST_TILE,)
    less_equal_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_LESS_EQUAL_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def less_equal_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def less_equal_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LESS_EQUAL_ SCALAR")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        if (
            A.dtype in (torch.float16, torch.float32)
            and numel
            >= _LESS_EQUAL_SCALAR_INPLACE_FAST_TILE
            * _LESS_EQUAL_SCALAR_INPLACE_MIN_GRID
            and numel % _LESS_EQUAL_SCALAR_INPLACE_FAST_TILE == 0
        ):
            return _less_equal_scalar_inplace_fast(A, float(B))
        less_equal_func_scalar_inplace(A, B, out0=A)
        return A
    return less_equal_func_scalar(A, B, out0=A)


_LESS_EQUAL_SCALAR_INPLACE_FAST_TILE = 131072
_LESS_EQUAL_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def less_equal_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _less_equal_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _LESS_EQUAL_SCALAR_INPLACE_FAST_TILE,)
    less_equal_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_LESS_EQUAL_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
