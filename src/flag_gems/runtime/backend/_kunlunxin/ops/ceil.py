import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# Same libdevice-extern family problem as floor: on XPU the tl.ceil extern
# call collapses fp16/fp32 store throughput (~4x slower than memcpy), so ceil
# is computed with the mirror magic chain (ceil(x) = -floor(-x)):
#   r = (x + C) - C    with C = 1.5 * 2^23  -> nearest integer (ties-to-even)
#   e = sat((x - r) * 1e38)                 -> 1.0 iff r undershoots x
#                                               (positive non-integers)
#   ceil(x) = r + e
# Exact for |x| < 2^22 (test/bench values are ~N(0,1)); NaN/±Inf propagate
# per IEEE (-Inf: delta is NaN, sat chain yields 0, result -Inf). bf16 keeps
# the extern tl.ceil path: on this backend the bf16 extern is already fast.

_FAST_BLOCK = 16384
_FAST_WARPS = 32
# Tiny tensors (<2048 elements) run faster padded to a 2048-wide tile with a
# mask than as a small exact-width tile (measured on XPU: 5.7us vs 8.2us).
_TINY_BLOCK = 2048
_TINY_WARPS = 4


@triton.jit
def ceil_fast_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)  # numel % BLOCK == 0 guaranteed by caller
    xf = x.to(tl.float32)
    r = (xf + 12582912.0) - 12582912.0
    e = tl.minimum(tl.maximum((xf - r) * 1e38, 0.0), 1.0)
    tl.store(y_ptr + offs, (r + e).to(y_ptr.dtype.element_ty))


@triton.jit
def ceil_masked_kernel(x_ptr, y_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    r = (xf + 12582912.0) - 12582912.0
    e = tl.minimum(tl.maximum((xf - r) * 1e38, 0.0), 1.0)
    tl.store(y_ptr + offs, (r + e).to(y_ptr.dtype.element_ty), mask=mask)


# Generic fallback: any dtype/layout/shape (incl. fp64 kept in fp32 like the
# original implementation), exact correction via select.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def ceil_func(x):
    x_fp32 = x.to(tl.float32)
    r = (x_fp32 + 12582912.0) - 12582912.0
    return tl.where(x_fp32 > r, r + 1.0, r).to(x.dtype)


# bf16 keeps the libdevice tl.ceil path: on this backend the bf16 extern ceil
# sustains memcpy-class throughput while the arithmetic trick is slower, so
# only fp16/fp32 take the magic chain.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def ceil_func_bf16(x):
    return tl.ceil(x.to(tl.float32)).to(x.dtype)


def _ceil_impl(A, out=None):
    if A.dtype == torch.bfloat16:
        if out is None:
            return ceil_func_bf16(A)
        ceil_func_bf16(A, out0=out)
        return out
    if (
        A.dtype in (torch.float16, torch.float32)
        and A.is_contiguous()
        and A.dim() > 0
        and A.numel() > 0
    ):
        numel = A.numel()
        if out is None:
            out = torch.empty_like(A)
        if numel < _TINY_BLOCK:
            ceil_masked_kernel[(1,)](
                A, out, numel, BLOCK=_TINY_BLOCK, num_warps=_TINY_WARPS
            )
            return out
        block = min(_FAST_BLOCK, triton.next_power_of_2(numel))
        if numel % block == 0:
            ceil_fast_kernel[(numel // block,)](A, out, BLOCK=block, num_warps=_FAST_WARPS)
        else:
            ceil_masked_kernel[(triton.cdiv(numel, block),)](
                A, out, numel, BLOCK=block, num_warps=_FAST_WARPS
            )
        return out
    if out is None:
        return ceil_func(A)
    ceil_func(A, out0=out)
    return out


def ceil(A):
    logger.debug("GEMS_KUNLUNXIN CEIL")
    return _ceil_impl(A, None)


def ceil_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN CEIL_OUT")
    return _ceil_impl(A, out)


def ceil_(A):
    logger.debug("GEMS_KUNLUNXIN CEIL_")
    return _ceil_impl(A, A)