import logging

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
    kunlunAutoGrid=False,
    unroll_num=8,
)


@triton.jit
def _fmod(x, y):
    # Correctly-rounded trunc-based fmod via exact FMA residual + bitwise off-by-one fix.
    #
    # q = trunc(div_rn(x, y)): the correctly-rounded fp32 quotient is truncated toward
    # zero. It differs from trunc(x/y) by +/-1 only when x/y sits within half an ulp of
    # an integer (~1e-5 per-lane rate). In that off-by-one case the exact residual
    # r = fma(-q, y, x) (= x - q*y, single-rounded) is tiny with sign(r) != sign(x).
    # fire (int -1/0) detects it: sign bits differ AND r != 0.
    # o = r + copysign(|y|, x) on fire gives the true remainder exactly.
    # All in int/bitwise ops: cheaper than the float compare/select family on this backend.
    q = tl.div_rn(x, y).to(tl.int32).to(tl.float32)
    r = tl.fma(-q, y, x)
    oi = r.to(tl.int32, bitcast=True)
    xi = x.to(tl.int32, bitcast=True)
    fire = ((oi ^ xi) >> 31) & ((oi | -oi) >> 31)
    cs_i = tl.abs(y).to(tl.int32, bitcast=True) | ((xi >> 31) << 31)
    return r + (cs_i & fire).to(tl.float32, bitcast=True)


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
