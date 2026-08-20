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

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    # isCloseMemoryAsync must stay at its default (True = async copy closed).
    # Enabling async copy (=False) makes ConvertTritonXPUToLLVM materialize a
    # multi-`ptr` async-buffer struct ON TOP of the buffer_size_limit=2048
    # i64-struct; both get re-printed on every insert/extractvalue, blowing the
    # compiled IR up to ~19GB (see benchmark/ir_dump/ir-logical_or-dev7.log) and
    # making the benchmark fail with ZeroDivisionError. unroll_num=8 + async
    # closed keeps the i64 buffer chunked so nothing explodes (same as bitwise_and).
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config_)
@triton.jit
def logical_or_func(x, y):
    return x.to(tl.int1).logical_or(y.to(tl.int1))


def logical_or(A, B):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_OR")
    return logical_or_func(A, B)


def logical_or_(A, B):
    # In-place variant was NOT overridden -> fell back to the generic
    # ops/logical_or.py path with the default pointwise_dynamic config (tile 512,
    # buffer_size_limit=2048, no unroll/autoGrid) -> catastrophic (60-1090ms on
    # large shapes, speedup ~0.001-0.4). Reuse the same tuned logical_or_func with
    # out0=A (same recipe as bitwise_and_tensor_).
    logger.debug("GEMS_KUNLUNXIN LOGICAL_OR_")
    logical_or_func(A, B, out0=A)
    return A
