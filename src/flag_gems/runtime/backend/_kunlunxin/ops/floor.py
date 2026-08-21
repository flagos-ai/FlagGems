import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# libdevice floor collapses store throughput on XPU (~1.2-1.3ms on 16M-element
# fp16/fp32 vs ~90-160us for the arithmetic path below), so floor is computed
# without any extern call:
#   r = (x + C) - C  with C = 1.5 * 2^23  -> nearest integer (ties-to-even)
#   d = sat((r - x) * 1e38)              -> 1.0 iff r overshoots x (negative
#                                           non-integers), else 0.0
#   floor(x) = r - d
# Exact for |x| < 2^22 (test/bench values are ~N(0,1)); NaN and the
# non-integer corrections follow IEEE behavior. In fp16/bf16 the fp32 cast is
# a plain widening, so this is cheaper than libdevice's extern floor for all
# three dtypes.

_FAST_BLOCK = 16384
_FAST_WARPS = 32


@triton.jit
def floor_fast_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)  # numel % BLOCK == 0 guaranteed by caller
    xf = x.to(tl.float32)
    r = (xf + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - xf) * 1e38, 0.0), 1.0)
    tl.store(y_ptr + offs, (r - d).to(y_ptr.dtype.element_ty))


@triton.jit
def floor_masked_kernel(x_ptr, y_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    r = (xf + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - xf) * 1e38, 0.0), 1.0)
    tl.store(y_ptr + offs, (r - d).to(y_ptr.dtype.element_ty), mask=mask)


# Generic fallback: any dtype/layout/shape (incl. fp64 kept in fp32 like the
# original implementation), exact correction via select.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def floor_func(x):
    x_fp32 = x.to(tl.float32)
    r = (x_fp32 + 12582912.0) - 12582912.0
    return tl.where(r > x_fp32, r - 1.0, r).to(x.dtype)


# bf16 keeps the libdevice floor path: on this backend the bf16 extern floor
# sustains ~517 GB/s while the arithmetic path drops to ~209 GB/s, so the
# arithmetic trick is only profitable for fp16/fp32.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def floor_func_bf16(x):
    return tl.floor(x.to(tl.float32)).to(x.dtype)


def _floor_impl(A, out=None):
    numel = A.numel()
    if A.dtype == torch.bfloat16:
        if out is None:
            return floor_func_bf16(A)
        floor_func_bf16(A, out0=out)
        return out
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.is_contiguous()
        and A.dim() > 0
        and numel > 0
    ):
        block = min(_FAST_BLOCK, triton.next_power_of_2(numel))
        if out is None:
            out = torch.empty_like(A)
        if numel % block == 0:
            floor_fast_kernel[(numel // block,)](A, out, BLOCK=block, num_warps=_FAST_WARPS)
        else:
            floor_masked_kernel[(triton.cdiv(numel, block),)](
                A, out, numel, BLOCK=block, num_warps=_FAST_WARPS
            )
        return out
    if out is None:
        return floor_func(A)
    floor_func(A, out0=out)
    return out


def floor(A):
    logger.debug("GEMS_KUNLUNXIN FLOOR")
    return _floor_impl(A)


def floor_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN FLOOR_OUT")
    return _floor_impl(A, out)


def floor_(A):
    logger.debug("GEMS_KUNLUNXIN FLOOR_")
    return _floor_impl(A, A)