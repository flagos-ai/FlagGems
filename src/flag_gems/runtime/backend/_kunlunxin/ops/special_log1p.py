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

# Kunlunxin (XPU) override of special_log1p / special_log1p.out.
#
# The generic `flag_gems.ops.special_log1p` runs through the *generic*
# `flag_gems.utils.pointwise_dynamic` codegen, which emits no XPU launch
# tuning (no kunlunAutoGrid, no unroll, no buffer_size_limit). On XPU that
# leaves the memory-bound log1p kernel catastrophically slow: measured
# gems latency on the comprehensive pointwise matrix is ~19.5 ms for
# [4096, 4096] (gems speedup ~0.014) -- see
# /ssd3/liuhanlin/tmp/perf/special_log1p/baseline.
#
# Fix: move the op onto the vendor pointwise_dynamic with a tuned
# CodeGenConfig. Config evolution (all single-variable, gated on the full
# benchmark + functional file, see solution/performance/special_log1p_perf_fix.md):
#   C1 log1p-recipe (unroll 8)            balanced 0.8505 (10.9x)
#   C2 unroll 8 -> 16                     balanced 0.8904 (kept)
#   C3 buffer_size_limit 4096 -> 8192     0.8699   (reverted, regression)
#   C4 isCloseMemoryAsync True -> False   0.8920   (reverted, +0.2% noise)
#   C5 unroll 16 -> 32                    0.8729   (reverted, fp32 -6%)
# with kunlunAutoGrid=False throughout (special_erfc round: the auto-grid path
# pins every shape with sum(shape) <= 2048*64 to a single CTA --
# pointwise_dynamic.py:884-896 -- which regresses small shapes and helps
# nothing on large ones).
#
# IMPORTANT (inherited from the log1p override): isCloseVectorization=True
# (vectorization CLOSED). With vectorization OPEN the vectorized log
# MISCOMPILES bf16: ~1.6% of elements come out off by exactly +ln(2)=0.6931.
# Kernel body (tl.log(1 + x_fp32)) is byte-identical to the generic op.
import logging

import torch
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
    isCloseVectorization=True,
    kunlunAutoGrid=False,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def special_log1p_func(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


def special_log1p(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG1P")
    if isinstance(A, torch.Tensor):
        return special_log1p_func(A)
    else:
        return torch.log(torch.tensor(A + 1.0))


def special_log1p_out(A, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG1P_OUT")
    return special_log1p_func(A, out0=out)
