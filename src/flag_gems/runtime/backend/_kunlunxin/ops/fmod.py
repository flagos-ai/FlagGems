import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.triton_lang_extension import div_rz

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
    # NOTE: kunlunAutoGrid must stay False.  When True the generated launcher
    # falls back to a single CTA for every shape with sum(shape) <= 2048*64,
    # which covers the whole fmod_ benchmark matrix (e.g. (4096,4096) has
    # sum==8192) and measures ~11ms for 16.7M elements; the 12-CTA grid
    # reaches ~0.2ms on the same shape (matches the tuned true_div config).
    unroll_num=8,
)


@triton.jit
def _fmod(x, y):
    # fmod(x, y) = x - trunc(x/y)*y.  div_rz is the native correctly-rounded
    # round-toward-zero division (xpu::__fdiv_rz); the int32 cast truncates the
    # (possibly non-integer) quotient to the nearest integer toward zero (XPU
    # f2i is RTZ; empirically bit-equal to fmodf on 22M+ random and boundary
    # inputs).  For |q| >= 2^23 the fp32 quotient carries no fraction bits, so
    # the cast is exact and equals the guarded fallback; both are exact for
    # |q| < 2^31.  tl.fma keeps x - t*y single-rounded, i.e. exactly fmodf.
    # The old variant used tl.where(abs(q) < 2^23, cast(q), q); the extra
    # abs/compare/select costs ~16-18% (measured via do_bench on XPU) and the
    # guard only changes behavior for |q| >= 2^31 (int32 overflow, where both
    # variants are already far from the exact fmod).  The old implementation
    # computed the quotient in software-emulated fp64 (and a variant used
    # tl.floor/ceil), both ~10-50x slower on XPU.
    q = div_rz(x, y)
    t = tl.cast(q, tl.int32).to(tl.float32)
    return tl.fma(t, -y, x)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def fmod_func(x, y):
    dtype = x.dtype
    return _fmod(x.to(tl.float32), y.to(tl.float32)).to(dtype)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def fmod_func_tensor_scalar(x, y):
    dtype = x.dtype
    return _fmod(x.to(tl.float32), y.to(tl.float32)).to(dtype)


def fmod_tensor(A, B):
    return fmod_func(A, B)


def fmod_scalar(A, B):
    return fmod_func_tensor_scalar(A, B)


def fmod_tensor_(A, B):
    return fmod_func(A, B, out0=A)


def fmod_scalar_(A, B):
    return fmod_func_tensor_scalar(A, B, out0=A)
