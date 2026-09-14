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
    # kunlunAutoGrid=True pins every pointwise shape with sum(shape) <=
    # 2048*64 to a single XPU CTA (see pointwise_dynamic gen_task_partition_1d).
    # xlogy benchmark shapes such as [4096,4096] (16M elements) and
    # [10000,65536] (655M elements) all fall in that bucket and were measured
    # 3-4x slower than an explicit 12-CTA grid; fixed 12 CTAs match it on the
    # remaining large shapes (same recipe as special_erfc).
    kunlunAutoGrid=False,
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


# In-place overloads. Without these vendor exports the registered
# ("xlogy_.Tensor", xlogy_) / ("xlogy_.Scalar_Other", xlogy_tensor_scalar_)
# entries in flag_gems/__init__.py keep binding the *generic*
# flag_gems.ops.xlogy_ wrappers, whose by-value import of the untuned
# flag_gems.ops.xlogy.xlogy_func never sees the vendor replacement
# (same SpecOpRegistrar name-based dispatch gap as divide, 2026-09-09).
# Measured on the xlogy_ comprehensive benchmark before this fix: 16M fp32
# elements took ~48.7 ms in gems (~0.7 GB/s, speedup 0.005) because the
# generic codegen carries no XPU tuning.
def xlogy_(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_")
    return xlogy_func(self, other, out0=self)


def xlogy_tensor_scalar_(self, other):
    logger.debug("GEMS_KUNLUNXIN XLOGY_TENSOR_SCALAR_")
    return xlogy_func_tensor_scalar(self, other, out0=self)
