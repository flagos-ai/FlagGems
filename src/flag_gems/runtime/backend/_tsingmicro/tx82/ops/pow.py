# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import triton
import triton.language as tl
from triton.backends.tsingmicro.precision import (
    restore_precision_mode,
    set_precision_mode,
)

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_TENSOR_TENSOR")
    precision_token = set_precision_mode("0")  # force to high-perf mode
    try:
        return pow_func(A, exponent)
    finally:
        restore_precision_mode(precision_token)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_TENSOR_TENSOR_")
    precision_token = set_precision_mode("0")  # force to high-perf mode
    try:
        return pow_func(A, exponent, out0=A)
    finally:
        restore_precision_mode(precision_token)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_TENSOR_SCALAR")
    precision_token = set_precision_mode("0")  # force to high-perf mode
    try:
        return pow_func_tensor_scalar(A, exponent)
    finally:
        restore_precision_mode(precision_token)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_TENSOR_SCALAR_")
    precision_token = set_precision_mode("0")  # force to high-perf mode
    try:
        return pow_func_tensor_scalar(A, exponent, out0=A)
    finally:
        restore_precision_mode(precision_token)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS_TSINGMICRO POW_SCALAR")
    # force set to high-perf mode
    precision_token = set_precision_mode("0")

    try:
        return pow_func_scalar_tensor(A, exponent)
    finally:
        restore_precision_mode(precision_token)
