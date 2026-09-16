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

# Two codegen variants for the two latency regimes on XPU:
# - small tensors (numel <= 4096): launch-bound -> 1-CTA (kunlunAutoGrid) path
#   keeps the whole tensor in one CTA.
# - medium/large tensors: 12-CTA grid-stride path with explicit unroll 16;
#   unroll 16 is the sweet spot measured for this op family on XPU
#   (sibling bitwise_left_shift_ closure, harness/solution/performance/).
config_small = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)
config_large = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_small)
@triton.jit
def bitwise_right_shift_small(a, b):
    return a >> b


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_large)
@triton.jit
def bitwise_right_shift_large(a, b):
    return a >> b


def _right_shift_kernel(a, b, *, out=None, out0=None):
    large = a.numel() > 4096
    if out is not None:
        if large:
            return bitwise_right_shift_large(a, b, out=out)
        return bitwise_right_shift_small(a, b, out=out)
    if out0 is not None:
        if large:
            return bitwise_right_shift_large(a, b, out0=out0)
        return bitwise_right_shift_small(a, b, out0=out0)
    if large:
        return bitwise_right_shift_large(a, b)
    return bitwise_right_shift_small(a, b)


def bitwise_right_shift(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN BITWISE_RIGHT_SHIFT")
    return _right_shift_kernel(self, other, out=out)


def bitwise_right_shift_(self, other, *, out=None):
    return _right_shift_kernel(self, other, out0=self)
