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
    kunlunAutoGrid=True,
    unroll_num=8,
)


@triton.jit
def _fmod(x, y):
    x64 = x.to(tl.float64)
    y64 = y.to(tl.float64)
    quotient = x64 / y64
    quotient = tl.where(quotient >= 0, tl.floor(quotient), -tl.floor(-quotient))
    return x64 - y64 * quotient


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def fmod_func(x, y):
    dtype = x.dtype
    return _fmod(x.to(tl.float32), y.to(tl.float32)).to(dtype)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_)
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
