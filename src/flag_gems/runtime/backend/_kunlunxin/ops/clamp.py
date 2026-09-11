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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# clamp_max is a pure memory-bound elementwise op (reads 1 tensor + scalar,
# writes 1) whose only compute is a min. Without a tuned config the default
# codegen emits a tiny 256-element tile with no unrolling, badly underutilizing
# the XPU (~1000x slower than torch: 49ms vs 0.04ms on 4096^2). Use div.py's
# tuned recipe (larger buffer + unroll) but keep vectorization OPEN
# (isCloseVectorization=False): this op is 1-in/1-out so wide vector DMA (esp.
# packing fp16/bf16) is the bandwidth lever. Measured on 4096^2: vec-open fp16
# 0.092ms vs vec-closed 0.155ms; [10000,65536] fp16 1.43ms vs 5.29ms. Unlike
# addcdiv (3-in, vec-closed best), always measure per op.
clamp_max_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def clamp_func_tensor(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min_tensor(x, mini):
    return tl.maximum(mini, x)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max_tensor(x, maxi):
    return tl.minimum(maxi, x)


def clamp_tensor(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLAMP_TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max_tensor(A, maxi)
    elif maxi is None:
        return clamp_func_min_tensor(A, mini)
    else:
        return clamp_func_tensor(A, mini, maxi)


def clamp_tensor_(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLAMP_TENSOR_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max_tensor(A, maxi, out0=A)
    elif maxi is None:
        return clamp_func_min_tensor(A, mini, out0=A)
    else:
        return clamp_func_tensor(A, mini, maxi, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clamp_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min(x, mini):
    return tl.maximum(mini, x)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max(x, maxi):
    return tl.minimum(maxi, x)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=clamp_max_config,
)
@triton.jit
def clamp_max_func(x, maxi):
    return tl.minimum(maxi, x)


def clamp_max(A, max_value):
    logger.debug("GEMS_KUNLUNXIN CLAMP_MAX")
    if max_value is None:
        raise ValueError("max_value must not be None")
    return clamp_max_func(A, max_value)


def clamp_max_(A, max_value):
    logger.debug("GEMS_KUNLUNXIN CLAMP_MAX_")
    if max_value is None:
        raise ValueError("max_value must not be None")
    return clamp_max_func(A, max_value, out0=A)


def clamp_min(A, mini):
    logger.debug("GEMS_KUNLUNXIN CLAMP_MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    return clamp_func_min(A, mini)


def clamp_min_(A, mini):
    logger.debug("GEMS_KUNLUNXIN CLAMP_MIN_")
    if mini is None:
        raise ValueError("Mini must not be None")
    return clamp_func_min(A, mini, out0=A)


def clamp(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLAMP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max(A, maxi)
    elif maxi is None:
        return clamp_func_min(A, mini)
    else:
        return clamp_func(A, mini, maxi)


def clamp_(A, mini=None, maxi=None):
    logger.debug("GEMS_KUNLUNXIN CLAMP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max(A, maxi, out0=A)
    elif maxi is None:
        return clamp_func_min(A, mini, out0=A)
    else:
        return clamp_func(A, mini, maxi, out0=A)
