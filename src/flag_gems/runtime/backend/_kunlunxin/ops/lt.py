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
import os

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
    isCloseMemoryAsync=False,
    unroll_num=8,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


def lt(A, B):
    logger.debug("GEMS_KUNLUNXIN LT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = lt_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR")
    res = lt_func_scalar(A, B)
    return res


# lt_ / lt_scalar_ are the in-place aliases of lt (out = (x < y) written back
# into x). They were NOT overridden by kunlunxin, so they fell to the generic
# ops/lt_.py -- a bare `@pointwise_dynamic` with NO CodeGenConfig -> discrete /
# launch-bound slow path. Baseline IR (ir-lt_-dev0 / ir-lt_scalar_-dev1) shows
# 512-wide discrete masked stores (`tt.ptr<f16, 0>` + per-element i32 column
# offsets) -> catastrophic latency on large shapes.
#
# Fix: a dedicated tuned pointwise_dynamic that writes in place (out0=A).
# CRITICAL: this in-place variant must NOT reuse lt's config_ -- config_ has
# `isCloseMemoryAsync=False` (async memory copy ON), and with in-place aliasing
# (input tensor == output tensor) the async double-buffered copy path deadlocks
# the device ("noc idle timeout" hang). The out-of-place lt is fine because its
# output is a fresh bool tensor (no aliasing). So use a config with the DEFAULT
# isCloseMemoryAsync (True = async closed), mirroring the proven in-place op
# greater_equal_. Body returns tl.where(...,1,0) (int 0/1) which stores cleanly
# into A's original fp16/bf16/fp32 dtype. The scalar path additionally must NOT
# set the TRITONXPU_COMPARE_FUSION / FP16_FAST fusion env vars (tensor-vs-scalar
# fp16 compare trips `arith.cmpf same-type` -> uni_sram overflow compile fail).
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config_inplace_)
@triton.jit
def lt_func_(x, y):
    return tl.where(x.to(tl.float32) < y.to(tl.float32), 1, 0)


def lt_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_")
    if A.device != B.device:
        B = B.to(A.device)
    lt_func_(A, B, out0=A)
    return A


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_inplace_,
)
@triton.jit
def lt_func_scalar_(x, y):
    return tl.where(x.to(tl.float32) < y, 1, 0)


def lt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR_")
    lt_func_scalar_(A, B, out0=A)
    return A
