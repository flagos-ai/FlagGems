import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.ne_ import ne_ as _generic_ne_
from flag_gems.ops.ne_ import ne_scalar_ as _generic_ne_scalar_

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


config_ = CodeGenConfig(
    512,
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
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logger.debug("GEMS_KUNLUNXIN NE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = ne_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y


def ne_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if A.is_contiguous() and dtype in (torch.float16, torch.float32, torch.bfloat16):
        s = float(B)
        wrapped = torch.tensor(s, dtype=dtype).item()
        if math.isfinite(wrapped):
            if (
                numel >= _NE_SCALAR_FAST_TILE * _NE_SCALAR_MIN_GRID
                and numel % _NE_SCALAR_FAST_TILE == 0
            ):
                return _ne_scalar_fast(
                    A, float(wrapped), (numel // _NE_SCALAR_FAST_TILE,)
                )
            if numel >= _NE_SCALAR_MASKED_MIN and numel % _NE_SCALAR_FAST_TILE != 0:
                return _ne_scalar_fast_masked(A, float(wrapped), numel)
    res = ne_func_scalar(A, B)
    return res


_NE_SCALAR_FAST_TILE = 131072
_NE_SCALAR_MIN_GRID = 128
_NE_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def ne_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    d = tl.abs(x - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t)


def _ne_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    ne_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_NE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out_bool = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out_bool, False)
    return out_bool


@triton.jit
def ne_scalar_fast_masked_kernel(out_ptr, y_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    y = tl.load(y_ptr + tid, mask=mask).to(tl.float32)
    d = tl.abs(y - scalar)
    t = tl.minimum(1.0, d * 1.0e30 * 1.0e15)
    tl.store(out_ptr + tid, t, mask=mask)


def _ne_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _NE_SCALAR_FAST_TILE),)
    ne_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_NE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out_bool = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out_bool, False)
    return out_bool


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
def ne_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    return t


def ne_(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_ TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel >= _NE_TENSOR_INPLACE_FAST_TILE * _NE_TENSOR_INPLACE_MIN_GRID
            and numel % _NE_TENSOR_INPLACE_FAST_TILE == 0
        ):
            return _ne_tensor_inplace_fast(A, B, numel)
        ne_func_tensor_inplace(A, B, out0=A)
        return A
    return _generic_ne_(A, B)


_NE_TENSOR_INPLACE_FAST_TILE = 131072
_NE_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def ne_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _ne_tensor_inplace_fast(A, B, numel):
    grid = (numel // _NE_TENSOR_INPLACE_FAST_TILE,)
    ne_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_NE_TENSOR_INPLACE_FAST_TILE,
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
def ne_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    return t


def ne_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_ SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        wrapped = float(torch.tensor(float(B), dtype=dtype).item())
        if math.isfinite(wrapped):
            if (
                dtype in (torch.float16, torch.float32)
                and numel >= _NE_SCALAR_INPLACE_FAST_TILE * _NE_SCALAR_INPLACE_MIN_GRID
                and numel % _NE_SCALAR_INPLACE_FAST_TILE == 0
            ):
                return _ne_scalar_inplace_fast(A, wrapped)
            ne_func_scalar_inplace(A, B, out0=A)
            return A
    return _generic_ne_scalar_(A, B)


_NE_SCALAR_INPLACE_FAST_TILE = 131072
_NE_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def ne_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _ne_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _NE_SCALAR_INPLACE_FAST_TILE,)
    ne_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_NE_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
