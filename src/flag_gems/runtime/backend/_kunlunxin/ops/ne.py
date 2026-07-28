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


# NOTE: `kunlunAutoGrid=True` + `unroll_num=8` are what make the sibling tuned
# comparison ops (gt / greater / greater_scalar) reach ~0.23-0.41 on large
# shapes. ne/ne_scalar previously shipped a bare config WITHOUT them and were
# stuck at ~0.14 (gems ~7.95ms vs torch ~1.08ms on the 65536-wide shapes, IR
# baseline `harness/perf_ir_3/ir-ne_scalar-dev3.log`). Adding the two params
# lifts throughput ~1.6x (mirrors greater_scalar, zero algorithm change).
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)


def ne(A, B):
    logger.debug("GEMS_KUNLUNXIN NE")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = ne_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def ne_func_scalar(x, y):
    return x.to(tl.float32) != y


def ne_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN NE_SCALAR")
    # Like gt_scalar / greater_scalar, the scalar path must NOT set
    # TRITONXPU_COMPARE_FUSION / TRITONXPU_FP16_FAST: for tensor-vs-scalar the
    # fusion env vars make the compiler emit an fp16 compare that trips
    # `arith.cmpf same-type` and overflows uni_sram -> compile failure.
    res = ne_func_scalar(A, B)
    return res
