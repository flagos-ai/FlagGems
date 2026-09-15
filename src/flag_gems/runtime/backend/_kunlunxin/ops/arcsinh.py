import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# arcsinh/arcsinh_/arcsinh_out were not overridden by kunlunxin, so they fell to
# the generic KernelGen ops/arcsinh.py: a hand-written kernel computing
# log(x + sqrt(x*x + 1)) in fp32 with a per-shape lambda grid. Problems on XPU:
# (1) correctness: that naive formula suffers catastrophic cancellation for
#     x < 0 at |x| >= ~1e6 (x + sqrt(x*x + 1) rounds to 0 in fp32 -> log(0)
#     = -inf), so arcsinh(-1e6) returned -inf instead of -14.5087 (native torch
#     on the DEVICE returns -14.5087, verified); arcsinh(-inf) returned NaN
#     instead of -inf; and for x > 0 the sign is fine but x*x overflows to +inf
#     at |x| > ~1.8e19;
# (2) the triton-XPU device asinhf intrinsic (tl_extra_shim.asinh ->
#     xpu/kernel/xtdk_trigonometric.h: asinh(x) = log(x + sqrt(1 + x*x)) is
#     implemented with the SAME naive formula, so it reproduces exactly the same
#     wrong results for x < 0 and cannot be used;
# (3) performance: the kernel recomputes asinh as log+sqrt+mul+add in fp32.
# Fix: mirror sibling unary ops (arcsin.py/arccos.py/cos.py) -- @pointwise_dynamic
# with the tuned config_ (vec-OPEN + kunlunAutoGrid + unroll8) -> shape-
# independent kernel (compiles ONCE) + contiguous block-DMA tiles, and use the
# numerically stable odd-symmetric formula
#     asinh(x) = sign(x) * log(|x| + sqrt(1 + |x|^2))
# For y = |x| >= 0 the sum (y + sqrt(1 + y*y)) always lies in [1, 2y], so there
# is NO cancellation and no precision loss (and for y > 512 the sum equals 2y to
# fp32 rounding, which is exactly right since asinh(y) = log(2y) up to 2^-24).
# Known limitation shared with the generic implementation (and the whole
# naive-intrinsic family): for |x| > ~1.8e19, y*y overflows to +inf and the
# result overflows to +-inf (true value is +-~89); native xdnn handles it but
# that is far outside any real tensor range.
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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def arcsinh_func(x):
    xf = x.to(tl.float32)
    y = tl.abs(xf)
    r = tl.log(y + tl.sqrt(1.0 + y * y))
    return tl.where(xf < 0, -r, r)


def arcsinh(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSINH FORWARD")
    if out is None:
        return arcsinh_func(x)
    arcsinh_func(x, out0=out)
    return out


def arcsinh_(x):
    logger.debug("GEMS_KUNLUNXIN ARCSINH INPLACE")
    arcsinh_func(x, out0=x)
    return x


def arcsinh_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ARCSINH OUT")
    return arcsinh(x, out=out)
