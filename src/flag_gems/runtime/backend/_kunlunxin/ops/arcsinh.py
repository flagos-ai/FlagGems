import logging

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
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def arcsinh_func(x):
    x32 = x.to(tl.float32)
    ax = tl.abs(x32)
    # Stable form (see asinh_ for the full derivation): for |x| > 1.84e19 the
    # x*x term overflows float32 and log(ax + sqrt(ax*ax+1)) silently returns
    # inf; the asymptotically equal 2*ax branch fixes it.  The single tl.log
    # sits outside the tl.where (a min/max-blend rewrite measured 5x worse:
    # XPU pointwise is ALU-bound).
    r = tl.where(
        ax > 1e16,
        ax + ax,
        ax + tl.sqrt(ax * ax + 1.0),
    )
    y = tl.log(r)
    return tl.where(x32 < 0.0, -y, y).to(x.dtype)


def arcsinh(A):
    return arcsinh_func(A)


def arcsinh_(A):
    arcsinh_func(A, out0=A)
    return A


def arcsinh_out(A, out):
    # ATen arcsinh on integer inputs produces a float32 result; the kernel
    # casts back to x.dtype, so promote an integer input to float32 here to
    # avoid a silently truncated (wrong) value being written into a float out.
    if not A.is_floating_point():
        A = A.to(torch.float32)
    return arcsinh_func(A, out0=out)
