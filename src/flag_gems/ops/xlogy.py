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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


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


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def xlogy_func(x, y):
    return _xlogy_compute(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def xlogy_func_tensor_scalar(x, y):
    return _xlogy_compute(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def xlogy_func_scalar_tensor(x, y):
    return _xlogy_compute(x, y)


def xlogy(self, other):
    logger.debug("GEMS XLOGY")
    return xlogy_func(self, other)


def xlogy_out(self, other, out):
    logger.debug("GEMS XLOGY_OUT")
    xlogy_func(self, other, out0=out)
    return out


def xlogy_tensor_scalar(self, other):
    logger.debug("GEMS XLOGY_TENSOR_SCALAR")
    return xlogy_func_tensor_scalar(self, other)


def xlogy_tensor_scalar_out(self, other, out):
    logger.debug("GEMS XLOGY_TENSOR_SCALAR_OUT")
    xlogy_func_tensor_scalar(self, other, out0=out)
    return out


def xlogy_scalar_tensor(self, other):
    logger.debug("GEMS XLOGY_SCALAR_TENSOR")
    return xlogy_func_scalar_tensor(self, other)


def xlogy_scalar_tensor_out(self, other, out):
    logger.debug("GEMS XLOGY_SCALAR_TENSOR_OUT")
    xlogy_func_scalar_tensor(self, other, out0=out)
    return out
