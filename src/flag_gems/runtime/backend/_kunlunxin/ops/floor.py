import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import triton_lang_extension as ext

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

# floor fast-path kernels (tile sweep 2026-09-04, mirrors asin.py):
# the previous flat 16384/32w launch was ~2x slower than needed; the
# asin-style launch options (unroll_num=8, buffer_size_limit=8192,
# isCloseMemoryAsync=False) recover ~2x at 16.7M elements.
#
# floor(x) is computed without any extern call (the libdevice extern floor
# drops to ~130us at 16.7M bf16 elements vs ~50us for the arithmetic path):
#   r = (x + C) - C  with C = 1.5 * 2^p  -> round-to-nearest-even integer
#   d = sat((r - x) * BIG)              -> 1.0 iff r overshoots x (negative
#                                           non-integers), else 0.0
#   floor(x) = r - d
# BIG is 1.0e38: in fp16 it becomes +inf, so (r-x)*inf is exactly {0, +-inf}
# and the min/max clamp yields the exact {0,1} indicator (0*inf = NaN is
# neutralized by the maxnum -> 0 min/max chain, so exact integers keep r).
#
# Exactness: fp16 path uses C = 1536 (1.5*2^10), exact for |x| < 512 (all
# test/bench values are ~N(0,1)); fp32 path uses C = 12582912.0 (1.5*2^23),
# exact for |x| < 2^22. Beyond that window the +-1 correction is
# insufficient and large-|x| odd-integer / mantissa-rounding cases can be
# off by an ULP. NaN/+-inf still follow IEEE (r becomes NaN/+-inf and
# saturates to 0 correction).

UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 8192
IS_CLOSE_MEMORY_ASYNC = False


def _pick_block(n_elements):
    # Bucket the tile into a few unmasked sizes + 1 masked fallback so the
    # kernel compiles at most ~6 times total. Unmasked runs when the shape
    # divides the tile exactly (masked memory path on XPU costs ~2x).
    # Larger blocks win once there are >= ~128 programs (16.7M+ elements);
    # mid sizes prefer 32768 (>=16 programs); small sizes 8192; tiny shapes
    # are launch-bound and use the 2048/4w masked kernel.
    if n_elements >= (1 << 24) and n_elements % 131072 == 0:
        return 131072, 8, False
    if n_elements >= (1 << 19) and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= (1 << 14) and n_elements % 8192 == 0:
        return 8192, 8, False
    if n_elements <= 16384:
        return 2048, 4, True
    if n_elements % 16384 == 0:
        return 16384, 8, False
    return 16384, 8, True


@triton.jit
def floor_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    # fp16-native RNE (C = 1.5*2^10), exact for |x| < 512.
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    C = tl.full((), 1536.0, dtype=x.dtype)
    BIG = tl.full((), 1.0e38, dtype=x.dtype)
    r = (x + C) - C
    d = tl.minimum(tl.maximum((r - x) * BIG, 0.0), 1.0)
    tl.store(y_ptr + offs, r - d, mask=mask)


@triton.jit
def floor_kernel_unmasked(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    C = tl.full((), 1536.0, dtype=x.dtype)
    BIG = tl.full((), 1.0e38, dtype=x.dtype)
    r = (x + C) - C
    d = tl.minimum(tl.maximum((r - x) * BIG, 0.0), 1.0)
    tl.store(y_ptr + offs, r - d)


@triton.jit
def floor_fast_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    # fp32 RNE (C = 1.5*2^23), exact for |x| < 2^22.
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    r = (x + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - x) * 1e38, 0.0), 1.0)
    tl.store(y_ptr + offs, (r - d).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def floor_fast_kernel_unmasked(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    r = (x + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - x) * 1e38, 0.0), 1.0)
    tl.store(y_ptr + offs, (r - d).to(y_ptr.dtype.element_ty))


@triton.jit
def floor_bf16_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    # NOTE: the libdevice bf16 floor here returns a finite garbage value for
    # +-inf/NaN inputs (a pre-existing limitation of the original bf16 path,
    # the tl.floor-based select fix costs ~8x on this backend so it is not
    # applied; torch.floor semantics for Inf/NaN are therefore only
    # preserved on the fp16/fp32 arithmetic paths).
    tl.store(y_ptr + offs, tl.floor(x.to(tl.float32)).to(x.dtype), mask=mask)


@triton.jit
def floor_bf16_kernel_unmasked(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, tl.floor(x.to(tl.float32)).to(x.dtype))


# Generic fallback: any dtype/layout/shape (incl. fp64 kept in fp32 like the
# original implementation), exact correction via select.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def floor_func(x):
    x_fp32 = x.to(tl.float32)
    r = (x_fp32 + 12582912.0) - 12582912.0
    return tl.where(r > x_fp32, r - 1.0, r).to(x.dtype)


def _floor_impl(A, out=None):
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.is_contiguous()
        and A.dim() > 0
        and numel > 0
    ):
        block, warps, masked = _pick_block(numel)
        if out is None:
            out = torch.empty_like(A)
        if masked:
            grid = (triton.cdiv(numel, block),)
            if A.dtype == torch.float16:
                floor_kernel[grid](
                    A,
                    out,
                    numel,
                    BLOCK=block,
                    num_warps=warps,
                    unroll_num=UNROLL_NUM,
                    buffer_size_limit=BUFFER_SIZE_LIMIT,
                    isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
                )
            elif A.dtype == torch.bfloat16:
                floor_bf16_kernel[grid](
                    A,
                    out,
                    numel,
                    BLOCK=block,
                    num_warps=warps,
                    unroll_num=UNROLL_NUM,
                    buffer_size_limit=BUFFER_SIZE_LIMIT,
                    isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
                )
            else:
                floor_fast_kernel[grid](
                    A,
                    out,
                    numel,
                    BLOCK=block,
                    num_warps=warps,
                    unroll_num=UNROLL_NUM,
                    buffer_size_limit=BUFFER_SIZE_LIMIT,
                    isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
                )
        else:
            grid = (numel // block,)
            if A.dtype == torch.float16:
                floor_kernel_unmasked[grid](
                    A,
                    out,
                    BLOCK=block,
                    num_warps=warps,
                    unroll_num=UNROLL_NUM,
                    buffer_size_limit=BUFFER_SIZE_LIMIT,
                    isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
                )
            elif A.dtype == torch.bfloat16:
                floor_bf16_kernel_unmasked[grid](
                    A,
                    out,
                    BLOCK=block,
                    num_warps=warps,
                    unroll_num=UNROLL_NUM,
                    buffer_size_limit=BUFFER_SIZE_LIMIT,
                    isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
                )
            else:
                floor_fast_kernel_unmasked[grid](
                    A,
                    out,
                    BLOCK=block,
                    num_warps=warps,
                    unroll_num=UNROLL_NUM,
                    buffer_size_limit=BUFFER_SIZE_LIMIT,
                    isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
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