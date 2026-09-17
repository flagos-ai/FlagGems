import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

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
def ge_func(x, y):
    return x.to(tl.float32) >= y


def ge(A, B):
    logger.debug("GEMS_KUNLUNXIN GE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = ge_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GE_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if A.is_contiguous() and dtype in (torch.float16, torch.float32, torch.bfloat16):
        s = float(B)
        if math.isfinite(s) and s == float(torch.tensor(s, dtype=dtype).item()):
            raw = _ge_scalar_raw(A, s, dtype)
            if raw is not None:
                return raw
            if numel >= _GE_SCALAR_FAST_TILE and numel % _GE_SCALAR_FAST_TILE == 0:
                return _ge_scalar_fast(A, s, (numel // _GE_SCALAR_FAST_TILE,))
            if numel >= _GE_SCALAR_MASKED_MIN and numel % _GE_SCALAR_FAST_TILE != 0:
                return _ge_scalar_fast_masked(A, s, numel)
    res = ge_func_scalar(A, B)
    return res


_GE_SCALAR_RAW_MIN = 1 << 24
_GE_SCALAR_MIN_NORM = 1.1754943508222875e-38


def _ge_scalar_raw(A, s, dtype):
    """ge(A, scalar) via the vendor comparison payloads, or None."""
    if A.numel() < _GE_SCALAR_RAW_MIN:
        return None
    from .greater import _raw_greater_scalar
    from .lt import _raw_lt_scalar

    if dtype == torch.float16:
        pred_s = float(
            torch.tensor(s, dtype=dtype)
            .nextafter(torch.tensor(float("-inf"), dtype=dtype))
            .item()
        )
        return _raw_greater_scalar(A, pred_s)
    if s == 0.0 or abs(s) >= _GE_SCALAR_MIN_NORM:
        s_eff = (
            -_GE_SCALAR_MIN_NORM
            if s == 0.0
            else float(
                torch.tensor(s, dtype=dtype)
                .nextafter(torch.tensor(float("-inf"), dtype=dtype))
                .item()
            )
        )
        return _raw_greater_scalar(A, s_eff)
    out = _raw_lt_scalar(A, s)
    if out is None:
        return None
    return torch.logical_not(out)


_GE_SCALAR_FAST_TILE = 131072
_GE_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def ge_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (scalar - x) * 1.0e38
    tl.store(out_ptr + tid, tl.maximum(0.0, 1.0 - t))


def _ge_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    ge_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def ge_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (scalar - x) * 1.0e38
    tl.store(out_ptr + tid, tl.maximum(0.0, 1.0 - t), mask=mask)


def _ge_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _GE_SCALAR_FAST_TILE),)
    ge_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_GE_SCALAR_FAST_TILE,
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
def greater_equal_func_(x, y):
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def greater_equal_(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_EQUAL_")
    if A.device != B.device:
        B = B.to(A.device)
    greater_equal_func_(A, B, out0=A)
    return A
