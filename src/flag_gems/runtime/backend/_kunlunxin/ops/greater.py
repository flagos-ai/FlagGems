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


config_scalar = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
    buffer_size_limit=8192,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def greater_func(x, y):
    return x.to(tl.float32) > y


def greater(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = greater_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


def greater_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_OUT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    if out is None:
        res = greater_func(A, B)
    else:
        greater_func(A, B, out0=out)
        res = out
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_scalar,
)
@triton.jit
def greater_func_scalar(x, y):
    return x.to(tl.float32) > y


def greater_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR")
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and (numel := A.numel()) >= _GREATER_SCALAR_FAST_TILE
        and numel % _GREATER_SCALAR_FAST_TILE == 0
        and numel // _GREATER_SCALAR_FAST_TILE >= _GREATER_SCALAR_MIN_GRID
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        return _greater_scalar_fast(A, float(B))
    res = greater_func_scalar(A, B)
    return res


_GREATER_SCALAR_FAST_TILE = 131072
_GREATER_SCALAR_MIN_GRID = 512


@triton.jit
def greater_scalar_fast_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    t = (x - scalar) * 1.0e30
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(out_ptr + tid, t)


def _greater_scalar_fast(A, scalar):
    out32 = torch.empty_like(A, dtype=torch.float32)
    grid = (A.numel() // _GREATER_SCALAR_FAST_TILE,)
    greater_scalar_fast_kernel[grid](
        out32,
        A,
        scalar,
        TILE=_GREATER_SCALAR_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    out = torch.empty_like(A, dtype=torch.bool)
    torch.ops.aten._copy_from(out32, out, False)
    return out


# Direct scalar compare vectorizes on XPU with TRITONXPU_COMPARE_FUSION=1 (the
# path greater's tensor-tensor kernel rides). TRITONXPU_FP16_FAST must stay off
# or the fp16 compare trips a TritonXPUDtypeConvert compile failure. fp16 uses a
# native vector compare against a tl.full constant; f32 a plain compare; bf16
# widens to fp16 losslessly to reach that fast path. See not_equal.py.
_GREATER_SCALAR_TILE_F32 = 65536
_GREATER_SCALAR_TILE_HALF = 131072
_GREATER_SCALAR_MIN_GRID = 8
_GREATER_SCALAR_MASKED_MIN = 1 << 20
_FP16_MAX = 65504.0


@triton.jit
def greater_scalar_cmp_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float32)
    tl.store(out_ptr + tid, x > scalar)


@triton.jit
def greater_scalar_cmp_masked_kernel(out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    tl.store(out_ptr + tid, x > scalar, mask=mask)


@triton.jit
def greater_scalar_half_kernel(out_ptr, x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid).to(tl.float16)
    yv = tl.full([TILE], scalar, tl.float16)
    tl.store(out_ptr + tid, x > yv)


@triton.jit
def greater_scalar_half_masked_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float16)
    yv = tl.full([TILE], scalar, tl.float16)
    tl.store(out_ptr + tid, x > yv, mask=mask)


_GREATER_SCALAR_LAUNCH_OPTS = dict(
    num_warps=4,
    buffer_size_limit=8192,
    unroll_num=16,
    isCloseMemoryAsync=False,
)


def _greater_set_compare_env():
    prev = (
        os.environ.get("TRITONXPU_COMPARE_FUSION"),
        os.environ.get("TRITONXPU_FP16_FAST"),
    )
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "0"
    return prev


def _greater_restore_compare_env(prev):
    for key, val in zip(("TRITONXPU_COMPARE_FUSION", "TRITONXPU_FP16_FAST"), prev):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _greater_scalar_out_cmp(A, scalar, out, tile, use_half, numel):
    prev = _greater_set_compare_env()
    try:
        if numel % tile == 0 and numel >= tile * _GREATER_SCALAR_MIN_GRID:
            grid = (numel // tile,)
            kernel = (
                greater_scalar_half_kernel if use_half else greater_scalar_cmp_kernel
            )
            kernel[grid](
                out.view(torch.uint8),
                A,
                scalar,
                TILE=tile,
                **_GREATER_SCALAR_LAUNCH_OPTS,
            )
        else:
            grid = (math.ceil(numel / tile),)
            kernel = (
                greater_scalar_half_masked_kernel
                if use_half
                else greater_scalar_cmp_masked_kernel
            )
            kernel[grid](
                out.view(torch.uint8),
                A,
                scalar,
                numel,
                TILE=tile,
                **_GREATER_SCALAR_LAUNCH_OPTS,
            )
    finally:
        _greater_restore_compare_env(prev)
    return out


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
    if (
        out is not None
        and out.dtype == torch.bool
        and A.is_contiguous()
        and out.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and (numel := A.numel()) >= _GREATER_SCALAR_MASKED_MIN
    ):
        wrapped = float(torch.tensor(float(B), dtype=A.dtype).item())
        if math.isfinite(wrapped):
            tile = (
                _GREATER_SCALAR_TILE_F32
                if A.dtype == torch.float32
                else _GREATER_SCALAR_TILE_HALF
            )
            use_half = A.dtype == torch.float16 or (
                A.dtype == torch.bfloat16 and abs(wrapped) <= _FP16_MAX
            )
            return _greater_scalar_out_cmp(A, wrapped, out, tile, use_half, numel)
    if out is None:
        res = greater_func_scalar(A, B)
    else:
        greater_func_scalar(A, B, out0=out)
        res = out
    return res
