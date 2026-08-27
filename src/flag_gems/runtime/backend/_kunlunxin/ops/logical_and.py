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
    # Without an explicit config the codegen falls back to buffer_size_limit=2048
    # (pointwise_dynamic.py). For this bool op ConvertTritonXPUToLLVM then
    # materializes a 2048-wide llvm.struct<(i64, ...)> that is re-printed on every
    # insert/extractvalue, blowing the compiled IR up to ~16GB (see
    # benchmark/ir_dump/ir-logical_and-dev6.log). unroll_num=8 breaks the buffer
    # into unrolled chunks so the monolithic struct is never formed; keep
    # isCloseMemoryAsync at its default (True) to avoid the async-copy explosion
    # documented for bitwise_and.
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config_)
@triton.jit
def logical_and_func(x, y):
    return x.to(tl.int1).logical_and(y.to(tl.int1))


def logical_and(A, B):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_AND")
    return logical_and_func(A, B)


# In-place variant. Without a kunlunxin override the ATen `logical_and_` falls
# back to the generic op (`ops/logical_and.py`, default buffer_size_limit=2048,
# no XPU tuning), which is catastrophic on XPU: every gm2lm/lm2gm is judged
# discrete (offsetState=-1) so large shapes run at ~0.001-0.003 speedup
# (60-1085ms). Reusing the same config_ as the out-of-place variant
# (unroll_num=8, kunlunAutoGrid, 1d-tile, async closed) restores block-DMA
# contiguous access. out0=A writes in place.
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config_)
@triton.jit
def logical_and_func_(x, y):
    return tl.where((x != 0) & (y != 0), 1, 0)


def logical_and_(A, B):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_AND_")
    logical_and_func_(A, B, out0=A)
    return A
