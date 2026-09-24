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
    unroll_num=8,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


def lt(A, B):
    logger.debug("GEMS_KUNLUNXIN LT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = lt_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and numel >= _LT_SCALAR_MASKED_MIN
    ):
        wrapped = float(torch.tensor(float(B), dtype=dtype).item())
        if math.isfinite(wrapped):
            tile = (
                _LT_SCALAR_TILE_F32 if dtype == torch.float32 else _LT_SCALAR_TILE_HALF
            )
            # fp16 has a native vector compare; a bf16 value within the fp16
            # range widens to fp16 losslessly (7-bit mantissa is a subset of
            # fp16's 10-bit) and preserves ordering, so route it there too. Only
            # safe when the scalar is fp16-finite, else it would round to +/-inf.
            use_half = dtype == torch.float16 or (
                dtype == torch.bfloat16 and abs(wrapped) <= _FP16_MAX
            )
            if numel % tile == 0 and numel >= tile * _LT_SCALAR_MIN_GRID:
                return _lt_scalar_fast(A, wrapped, tile, (numel // tile,), use_half)
            return _lt_scalar_fast_masked(A, wrapped, tile, numel, use_half)
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        B = float(torch.tensor(float(B), dtype=dtype).item())
    res = lt_func_scalar(A, B)
    return res


# Direct scalar compare vectorizes on XPU with TRITONXPU_COMPARE_FUSION=1 (the
# path lt's tensor-tensor kernel rides). TRITONXPU_FP16_FAST must stay off, or
# the fp16 compare trips a TritonXPUDtypeConvert compile failure. fp16 uses a
# native vector compare against a tl.full constant; f32 a plain compare; bf16
# widens to fp16 to reach that fast path. See not_equal.py for the same recipe.
_LT_SCALAR_TILE_F32 = 65536
_LT_SCALAR_TILE_HALF = 131072
_LT_SCALAR_MIN_GRID = 8
_LT_SCALAR_MASKED_MIN = 1 << 20
_FP16_MAX = 65504.0


@triton.jit
def lt_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    tl.store(out_ptr + tid, x < scalar)


@triton.jit
def lt_scalar_fast_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    tl.store(out_ptr + tid, x < scalar, mask=mask)


@triton.jit
def lt_scalar_half_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float16)
    yv = tl.full([TILE], scalar, tl.float16)
    tl.store(out_ptr + tid, x < yv)


@triton.jit
def lt_scalar_half_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float16)
    yv = tl.full([TILE], scalar, tl.float16)
    tl.store(out_ptr + tid, x < yv, mask=mask)


def _lt_set_compare_env():
    prev = (
        os.environ.get("TRITONXPU_COMPARE_FUSION"),
        os.environ.get("TRITONXPU_FP16_FAST"),
    )
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "0"
    return prev


def _lt_restore_compare_env(prev):
    for key, val in zip(("TRITONXPU_COMPARE_FUSION", "TRITONXPU_FP16_FAST"), prev):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _lt_scalar_fast(A, scalar, tile, grid, use_half):
    out = torch.empty_like(A, dtype=torch.bool)
    kernel = lt_scalar_half_kernel if use_half else lt_scalar_fast_kernel
    prev = _lt_set_compare_env()
    try:
        kernel[grid](
            out.view(torch.uint8),
            A,
            scalar,
            TILE=tile,
            **_LT_SCALAR_LAUNCH_OPTS,
        )
    finally:
        _lt_restore_compare_env(prev)
    return out


def _lt_scalar_fast_masked(A, scalar, tile, numel, use_half):
    out = torch.empty_like(A, dtype=torch.bool)
    grid = (math.ceil(numel / tile),)
    kernel = lt_scalar_half_masked_kernel if use_half else lt_scalar_fast_masked_kernel
    prev = _lt_set_compare_env()
    try:
        kernel[grid](
            out.view(torch.uint8),
            A,
            scalar,
            numel,
            TILE=tile,
            **_LT_SCALAR_LAUNCH_OPTS,
        )
    finally:
        _lt_restore_compare_env(prev)
    return out


_LT_SCALAR_LAUNCH_OPTS = dict(
    num_warps=4,
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseMemoryAsync=False,
)


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
def lt_func_(x, y):
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def lt_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_")
    if A.device != B.device:
        B = B.to(A.device)
    lt_func_(A, B, out0=A)
    return A


config_scalar_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
    buffer_size_limit=4096,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_scalar_inplace_,
)
@triton.jit
def lt_func_scalar_(x, y):
    t = (y - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return t


def lt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR_")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32)
        and numel >= _LT_SCALAR_INPLACE_FAST_TILE
        and numel % _LT_SCALAR_INPLACE_FAST_TILE == 0
        and numel // _LT_SCALAR_INPLACE_FAST_TILE >= _LT_SCALAR_INPLACE_MIN_GRID
        and float(B) == 0.0
    ):
        return _lt_scalar_inplace_fast(A)
    if A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        B = float(torch.tensor(float(B), dtype=A.dtype).item())
    lt_func_scalar_(A, B, out0=A)
    return A


_LT_SCALAR_INPLACE_FAST_TILE = 131072
_LT_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def lt_scalar_inplace_fast_kernel(x_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (0.0 - x) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, t)


def _lt_scalar_inplace_fast(A):
    grid = (A.numel() // _LT_SCALAR_INPLACE_FAST_TILE,)
    lt_scalar_inplace_fast_kernel[grid](
        A,
        TILE=_LT_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


def less_(A, B):
    return lt_(A, B)


def less_scalar_(A, B):
    return lt_scalar_(A, B)
