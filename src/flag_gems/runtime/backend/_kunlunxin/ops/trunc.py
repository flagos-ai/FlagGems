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

_FAST_BLOCK = 16384
_FAST_WARPS = 32


# Round-towards-zero of an fp32 value, computed on its bit pattern by clearing
# the mantissa bits below the binary point.
#
# Integer-only on purpose, for two independent platform reasons:
#   * the libdevice `truncf` extern silently re-lowers its argument as an
#     integer conversion as soon as that argument is derived from a bf16 load,
#     mangling +-inf, NaN and -0.0 (`trunc(-inf) -> -2**31`, `trunc(inf)` and
#     `trunc(nan)` -> 0x4eff0000, `trunc(-0.5) -> +0.0`);
#   * every fp op of this backend behaves as "no signed zero", so no
#     arithmetic expression can return the -0.0 that `trunc(-0.5)` /
#     `trunc(-0.0)` owe: `-m`, `m * -1.0` and even the bare `-0.0` literal all
#     come back as +0.0 once the tile is vectorized wide.  Only the sign bit
#     can carry it, i.e. the mask below.
# Masking is also the cheapest of the three: 0.52 ms vs 0.37 ms for the
# (sign-of-zero-incorrect) float magic and 3.8 ms for `f32 -> i32 -> f32`
# bitcasting a computed value, on a 16.7M element fp32 tile.
#
#   exp (biased) >= 127 : n = 150 - exp fraction bits to clear (0 when the
#                         value is already integral)
#   exp           < 127 : n = 31 -> |x| < 1 truncates to +-0.0, and subnormal
#                         inputs / +-0.0 keep their sign this way
#   exp          == 255 : n = 0 -> +-inf and NaN pass through untouched
@triton.jit
def _trunc_fp32(xf):
    xi = xf.to(tl.int32, bitcast=True)
    exp = (xi >> 23) & 255
    n = tl.where(exp < 127, 31, tl.maximum(150 - exp, 0))
    return (xi & (-1 << n)).to(tl.float32, bitcast=True)


@triton.jit
def trunc_fast_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)  # numel % BLOCK == 0 guaranteed by caller
    t = _trunc_fp32(x.to(tl.float32))
    tl.store(y_ptr + offs, t.to(y_ptr.dtype.element_ty))


@triton.jit
def trunc_masked_kernel(x_ptr, y_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    t = _trunc_fp32(x.to(tl.float32))
    tl.store(y_ptr + offs, t.to(y_ptr.dtype.element_ty), mask=mask)


# Generic path: any dtype/layout/shape.
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def trunc_func(x):
    return _trunc_fp32(x.to(tl.float32)).to(x.dtype)


def _trunc_impl(A, out=None):
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32)
        and A.is_contiguous()
        and A.dim() > 0
        and numel
        > 1024  # tiny tensors: generic pointwise path is at/below the launch floor
    ):
        block = min(_FAST_BLOCK, triton.next_power_of_2(numel))
        if out is None:
            out = torch.empty_like(A)
        if numel % block == 0:
            trunc_fast_kernel[(numel // block,)](
                A, out, BLOCK=block, num_warps=_FAST_WARPS
            )
        else:
            trunc_masked_kernel[(triton.cdiv(numel, block),)](
                A, out, numel, BLOCK=block, num_warps=_FAST_WARPS
            )
        return out
    if out is None:
        return trunc_func(A)
    trunc_func(A, out0=out)
    return out


def trunc(A):
    logger.debug("GEMS_KUNLUNXIN TRUNC")
    return _trunc_impl(A)


def trunc_(A):
    logger.debug("GEMS_KUNLUNXIN TRUNC_")
    trunc_func(A, out0=A)
    return A
