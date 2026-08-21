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

# Tuned CodeGenConfig for the XPU pointwise codegen (memory/ALU bound unary op):
# - buffer_size_limit=8192 / unroll_num=16: max DMA chunk streaming throughput
#   (swept 4096..16384 x 8..32; 8192/16 is the plateau optimum).
# - isCloseVectorization=True: REQUIRED for XPU logarithm correctness (vectorization
#   closed; open vectorization miscompiles log-family kernels, see log1p/log_).
# - Two launch variants are used because the XPU pointwise grid strategy has a
#   per-shape optimum (kernel-level do_bench on the official matrix, probe evidence):
#     * single_CTA config (kunlunAutoGrid=True): 1 CTA for small / very large tiles
#       (the auto-grid "small shape" branch) -- best for tiny tensors and for the
#       huge 2M+ lane tiles (fp32/bf16 win, fp16 close).
#     * multi_CTA config (kunlunAutoGrid=False): always 12 CTAs (XPU BLOCK_NUM) --
#       best for the 8K..128K-element mid range (e.g. (1024,16), (64,64,16)) where
#       the single giant tile serializes.
#   The wrapper picks the variant by numel: 8192 < numel <= 131072 -> multi_CTA,
#   otherwise single_CTA (matches the current 1-vs-12 CTA behavior for all other
#   shapes, i.e. behavior is unchanged outside the mid range).
config_single_cta = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)
config_multi_cta = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=False,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=config_single_cta)
@triton.jit
def log_func_single(x):
    return tl.log(x.to(tl.float32))


# The two scalar fns must have distinct source text so the pointwise codegen
# cache keys do not collide inside one process (two wrappers, same pid).
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")], config=config_multi_cta)
@triton.jit
def log_func_multi(x):
    return tl.log(1.0000000000000000 * x.to(tl.float32))


# 12-CTA mode is beneficial only inside this numel window (probe: 2026-08-18);
# outside it the single-CTA mode is at least as fast.
_MULTI_CTA_MIN_NUMEL = 8192
_MULTI_CTA_MAX_NUMEL = 131072


def log(A):
    logger.debug("GEMS_KUNLUNXIN LOG")
    numel = A.numel()
    if _MULTI_CTA_MIN_NUMEL < numel <= _MULTI_CTA_MAX_NUMEL:
        return log_func_multi(A)
    return log_func_single(A)