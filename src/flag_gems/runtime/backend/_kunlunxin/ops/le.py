import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.le_ import le_ as _generic_le_
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
def le_func(x, y):
    return x.to(tl.float32) <= y


def le(A, B):
    logger.debug("GEMS_KUNLUNXIN LE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = le_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def le_func_scalar(x, y):
    return x.to(tl.float32) <= y


def le_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LE_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        s = float(B)
        if math.isfinite(s):
            raw = _le_scalar_raw(A, s, dtype)
            if raw is not None:
                return raw
        if numel >= _LE_SCALAR_FAST_TILE and numel % _LE_SCALAR_FAST_TILE == 0:
            return _le_scalar_fast(A, s, (numel // _LE_SCALAR_FAST_TILE,))
        if numel >= _LE_SCALAR_MASKED_MIN and numel % _LE_SCALAR_FAST_TILE != 0:
            return _le_scalar_fast_masked(A, s, numel)
    res = le_func_scalar(A, B)
    return res


_LE_SCALAR_RAW_MIN = 1 << 24
_LE_SCALAR_MIN_NORM = 1.1754943508222875e-38


def _le_scalar_raw(A, s, dtype):
    """le(A, scalar) via the vendor single-pass payload, or None."""
    if A.numel() < _LE_SCALAR_RAW_MIN:
        return None
    from .lt import _raw_lt_scalar

    if dtype == torch.float16:
        pred_s = float(
            torch.tensor(s, dtype=dtype)
            .nextafter(torch.tensor(float("inf"), dtype=dtype))
            .item()
        )
        return _raw_lt_scalar(A, pred_s)
    if s == 0.0 or abs(s) >= _LE_SCALAR_MIN_NORM:
        s_eff = (
            _LE_SCALAR_MIN_NORM
            if s == 0.0
            else float(
                torch.tensor(s, dtype=dtype)
                .nextafter(torch.tensor(float("inf"), dtype=dtype))
                .item()
            )
        )
        return _raw_lt_scalar(A, s_eff)
    return None


_LE_SCALAR_FAST_TILE = 131072
_LE_SCALAR_MASKED_MIN = 1 << 20


@triton.jit
def le_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e38
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, 1.0 - t)


def _le_scalar_fast(A, scalar, grid):
    out32 = torch.empty_like(A, dtype=torch.float32)
    le_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_LE_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


@triton.jit
def le_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    t = (x - scalar) * 1.0e38
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, 1.0 - t, mask=mask)


def _le_scalar_fast_masked(A, scalar, numel):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (math.ceil(numel / _LE_SCALAR_FAST_TILE),)
    le_scalar_fast_masked_kernel[grid](
        out32,
        A,
        scalar,
        numel,
        TILE=_LE_SCALAR_FAST_TILE,
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
def le_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def le_(A, B):
    logger.debug("GEMS_KUNLUNXIN LE_ TENSOR")
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
            and numel >= _LE_TENSOR_INPLACE_FAST_TILE * _LE_TENSOR_INPLACE_MIN_GRID
            and numel % _LE_TENSOR_INPLACE_FAST_TILE == 0
        ):
            return _le_tensor_inplace_fast(A, B, numel)
        le_func_tensor_inplace(A, B, out0=A)
        return A
    return _generic_le_(A, B)


_LE_TENSOR_INPLACE_FAST_TILE = 131072
_LE_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def le_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _le_tensor_inplace_fast(A, B, numel):
    grid = (numel // _LE_TENSOR_INPLACE_FAST_TILE,)
    le_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_LE_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
