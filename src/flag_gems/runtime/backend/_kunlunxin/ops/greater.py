import logging
import os

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


# --------------------------------------------------------------------------
# Scalar (tensor-vs-scalar) compare.
#
# Two independent compile-time levers, both measured with a *fresh*
# TRITON_CACHE_DIR (12-CTA rank-1 geometry, unroll_num=16,
# buffer_size_limit=8192, [4096,4096] fp16):
#
#   1) TRITONXPU_COMPARE_FUSION=1 alone: 0.336 ms -> 0.043 ms (~7.8x). Without
#      it the XPU compiler does not fuse the compare into the vectorized
#      load/store pipeline. Note TRITONXPU_FP16_FAST must NOT be set together
#      with it on this kernel -- the combination puts the compare back on the
#      slow path (0.336 ms), i.e. the tensor-path env recipe is wrong here.
#   2) compare dtype: for 16-bit floats, comparing in the tensor's own dtype
#      instead of promoting to fp32 gives another 0.059 -> 0.043 ms on fp16
#      (speedup 0.60 -> 0.83). This matches PyTorch semantics: a python-number
#      scalar is a "wrapped number" and does not participate in type promotion,
#      so torch casts the scalar down to the tensor dtype. Non-16-bit dtypes
#      keep the fp32 compare, which is required for correct int-tensor vs
#      float-scalar promotion.
#
# The former `_greater_scalar_fast` kernel (sub/mul/clamp trick on a fp32
# scratch buffer) was removed: on the very shapes it guarded (numel >= 131072,
# grid >= 512, contiguous) it measured 3.1-3.2x slower than this fused
# pointwise path ((268435456,) fp16: 1.696 ms vs 0.538 ms), so keeping it
# could not reach the >= 0.8 dtype-equal-weight Gems Speedup target.
# --------------------------------------------------------------------------
@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_scalar,
)
@triton.jit
def greater_func_scalar(x, y):
    if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        return x > y.to(x.dtype)
    else:
        return x.to(tl.float32) > y


def _scalar_fusion_env():
    """Enable the XPU compare-fusion pass for the scalar kernel compile.

    Returns the previous value so it can be restored (unlike the tensor path we
    must not unconditionally `del`, since the tensor path may be active in an
    enclosing frame).
    """
    prev = os.environ.get("TRITONXPU_COMPARE_FUSION")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    return prev


def _restore_scalar_fusion_env(prev):
    if prev is None:
        os.environ.pop("TRITONXPU_COMPARE_FUSION", None)
    else:
        os.environ["TRITONXPU_COMPARE_FUSION"] = prev


def greater_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR")
    prev = _scalar_fusion_env()
    try:
        return greater_func_scalar(A, B)
    finally:
        _restore_scalar_fusion_env(prev)


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GREATER_SCALAR_OUT")
    prev = _scalar_fusion_env()
    try:
        if out is None:
            return greater_func_scalar(A, B)
        greater_func_scalar(A, B, out0=out)
        return out
    finally:
        _restore_scalar_fusion_env(prev)
