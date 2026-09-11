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
    # tl.log with vectorization ON crashes the XPU ELF compiler (make_elf
    # Aborted); the proven tl.log recipe (see log1p.py) closes vectorization.
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@triton.jit
def _xlogy_compute(x, y):
    # Follows PyTorch aten semantics (in this precedence):
    #   NaN if y is NaN; 0 if x == 0; otherwise x * log(y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    y_is_nan = y_f32 != y_f32
    prod = x_f32 * tl.log(y_f32)
    res = tl.where(x_f32 == 0.0, 0.0, prod)
    return tl.where(y_is_nan, float("nan"), res)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def xlogy_func(x, y):
    return _xlogy_compute(x, y)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def xlogy_func_tensor_scalar(x, y):
    return _xlogy_compute(x, y)


@pointwise_dynamic(
    is_tensor=[False, True],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def xlogy_func_scalar_tensor(x, y):
    return _xlogy_compute(x, y)


def xlogy(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY")
    return xlogy_func(self, other)


def xlogy_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_OUT")
    xlogy_func(self, other, out0=out)
    return out


def xlogy_tensor_scalar(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR")
    return xlogy_func_tensor_scalar(self, other)


def xlogy_tensor_scalar_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR_OUT")
    xlogy_func_tensor_scalar(self, other, out0=out)
    return out


def xlogy_scalar_tensor(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_SCALAR_TENSOR")
    return xlogy_func_scalar_tensor(self, other)


def xlogy_scalar_tensor_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN XLOGY_SCALAR_TENSOR_OUT")
    xlogy_func_scalar_tensor(self, other, out0=out)
    return out
