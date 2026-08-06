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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def bitwise_xor_func(x, y):
    return x ^ y


def bitwise_xor_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_TENSOR")
    return bitwise_xor_func(A, B)


def bitwise_xor_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_TENSOR_")
    return bitwise_xor_func(A, B, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def bitwise_xor_func_scalar(x, y):
    return x ^ y


def bitwise_xor_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_SCALAR")
    return bitwise_xor_func_scalar(A, B)


def bitwise_xor_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_SCALAR_")
    return bitwise_xor_func_scalar(A, B, out0=A)


def bitwise_xor_scalar_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_SCALAR_TENSOR")
    return bitwise_xor_func_scalar(B, A)


xor = bitwise_xor_tensor
xor_ = bitwise_xor_tensor_
xor_scalar = bitwise_xor_scalar
xor_scalar_ = bitwise_xor_scalar_
xor_scalar_tensor = bitwise_xor_scalar_tensor
