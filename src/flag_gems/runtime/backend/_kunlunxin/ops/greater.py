import logging
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


def _greater_scalar_out_fast(A, scalar, out):
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
    torch.ops.aten._copy_from(out32, out, False)
    return out


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
    if (
        out is not None
        and A.is_contiguous()
        and out.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and (numel := A.numel()) >= _GREATER_SCALAR_FAST_TILE
        and numel % _GREATER_SCALAR_FAST_TILE == 0
        and numel // _GREATER_SCALAR_FAST_TILE >= _GREATER_SCALAR_MIN_GRID
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        return _greater_scalar_out_fast(A, float(B), out)
    if out is None:
        res = greater_func_scalar(A, B)
    else:
        greater_func_scalar(A, B, out0=out)
        res = out
    return res
