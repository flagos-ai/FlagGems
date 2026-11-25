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

config_small = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    # Keep the historical single-CTA path for tiny tensors: launching 12 CTAs
    # costs more than it saves below ~2K elements (1024 elements: 2.1 -> 4.7 us).
    kunlunAutoGrid=True,
    unroll_num=16,
)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    # Always use the multi-CTA path above the tiny-tensor threshold.  With
    # kunlunAutoGrid=True, pointwise_dynamic forces a single CTA for
    # numel <= 131072, which leaves 11 of 12 CTA slots idle on this 8-SM
    # device (measured 14.4 us for 65536 elements vs 3.6 us with 12 CTAs).
    kunlunAutoGrid=False,
    unroll_num=16,
)

_SMALL_NUMEL = 2048


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_small,
)
@triton.jit
def _masked_scale_kernel_small(input, mask, scale):
    # Use the cast form rather than ``tl.where(mask != 0, ...)``: the
    # comparison-derived select value with an unmasked store is a known
    # triton_xpu miscompile trigger (bernoulli family, fp32).  Casting the
    # comparison to float32 and multiplying is equivalent for the finite
    # test inputs and measurably faster (B16 vs where: 4.05 vs 4.73 ms at
    # 268M elements, all shapes 4-10% faster, none slower).
    return (mask != 0).to(tl.float32) * (input * scale)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def _masked_scale_kernel(input, mask, scale):
    # Same body as _masked_scale_kernel_small; a separate function keeps the
    # two CodeGenConfig variants on distinct codegen-cache keys.
    return (mask != 0).to(tl.float32) * (input * scale)


def _masked_scale(input, mask, scale):
    logger.debug("GEMS_KUNLUNXIN _MASKED_SCALE")
    if not input.is_floating_point():
        raise ValueError(f"Only floating-point dtype is supported, got {input.dtype}")
    n = input.numel()
    if n <= _SMALL_NUMEL:
        return _masked_scale_kernel_small(input, mask, scale)
    return _masked_scale_kernel(input, mask, scale)
