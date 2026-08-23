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
    # ALWAYS_BOOL output -> keep isCloseMemoryAsync at its DEFAULT (True = async
    # copy closed). Opening async (=False) on a bool op re-introduces the
    # ConvertTritonXPUToLLVM giant-struct explosion on top of buffer_size_limit
    # (same trap as logical_or/and/xor).
    #
    # buffer_size_limit=4096 + unroll_num=16 keeps the per-unroll chunk at
    # 4096/16 == 256 elements (identical to the previous 2048/8 == 256, so the
    # monolithic-struct explosion is still avoided) but doubles the in-flight
    # buffer, which measurably raises large-shape DMA throughput: on XPU the
    # big shapes ([4096,4096], [1024,65536], [64,64,4096]) get a stable ~8%
    # latency drop across fp16/fp32/bf16 vs the old unroll_num=8 / default
    # (2048) buffer. Small shapes stay launch-floor bound (unchanged).
    kunlunAutoGrid=True,
    unroll_num=16,
    buffer_size_limit=4096,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=config_)
@triton.jit
def logical_not_func(x):
    return not x.to(tl.int1)


def logical_not(A):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_NOT")
    return logical_not_func(A)


def logical_not_(A):
    # In-place variant was NOT overridden -> fell back to the generic
    # ops/logical_not.py path with the default pointwise_dynamic config (tile 512,
    # buffer_size_limit=2048, no unroll/autoGrid) -> catastrophic (38-161ms on
    # large shapes, speedup ~0.001-0.005). Reuse the now-tuned logical_not_func
    # with out0=A (same recipe as bitwise_not_).
    logger.debug("GEMS_KUNLUNXIN LOGICAL_NOT_")
    logical_not_func(A, out0=A)
    return A
