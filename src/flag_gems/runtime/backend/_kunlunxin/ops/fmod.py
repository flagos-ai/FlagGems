import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

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
    # xpu libdevice fmod(f32) is exactly-rounded (IEEE-754), avoiding both the
    # slow f64 software-emulation path and the fcmp-select truncation (known
    # slow/miscompile family on this backend).
    return tl_extra_shim.fmod(x, y)


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
