import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic
from .greater import _RAW_SMALL_SCALAR_LIMIT, _raw_greater_scalar

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
def gt_func(x, y):
    return x.to(tl.float32) > y


def gt(A, B):
    logger.debug("GEMS_KUNLUNXIN GT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = gt_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def gt_func_scalar(x, y):
    return x.to(tl.float32) > y


def gt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_SCALAR")
    if A.numel() >= _RAW_SMALL_SCALAR_LIMIT:
        raw_out = _raw_greater_scalar(A, B)
        if raw_out is not None:
            return raw_out
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if (
            numel >= _GT_SCALAR_FAST_TILE * _GT_SCALAR_MIN_GRID
            and numel % _GT_SCALAR_FAST_TILE == 0
        ):
            return _gt_scalar_fast(A, float(B), (numel // _GT_SCALAR_FAST_TILE,))
        if numel >= _GT_SCALAR_MASKED_MIN and numel % _GT_SCALAR_FAST_TILE != 0:
            return _gt_scalar_fast_masked(A, float(B), numel)
    res = gt_func_scalar(A, B)
    return res


_GT_SCALAR_FAST_TILE = 131072
_GT_SCALAR_MIN_GRID = 128
_GT_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def gt_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t)


def _gt_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    gt_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GT_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def gt_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (x - scalar) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t, mask=mask)


def _gt_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _GT_SCALAR_FAST_TILE),)
    gt_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_GT_SCALAR_FAST_TILE,
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


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def gt_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def gt_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def gt_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_ TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    gt_func_tensor_inplace(A, B, out0=A)
    return A


def gt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN GT_ SCALAR")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        if (
            A.dtype in (torch.float16, torch.float32)
            and numel >= _GT_SCALAR_INPLACE_FAST_TILE * _GT_SCALAR_INPLACE_MIN_GRID
            and numel % _GT_SCALAR_INPLACE_FAST_TILE == 0
        ):
            return _gt_scalar_inplace_fast(A, float(B))
        gt_func_scalar_inplace(A, B, out0=A)
        return A
    return gt_func_scalar(A, B, out0=A)


_GT_SCALAR_INPLACE_FAST_TILE = 131072
_GT_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def gt_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _gt_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _GT_SCALAR_INPLACE_FAST_TILE,)
    gt_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_GT_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
