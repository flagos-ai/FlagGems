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

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Two codegen variants for the two latency regimes on XPU:
# - small tensors (numel <= 4096): launch-bound -> 1-CTA (kunlunAutoGrid) path
#   keeps the whole tensor in one CTA.
# - medium/large tensors: 12-CTA grid-stride path with explicit unroll 16;
#   unroll 16 is the sweet spot measured for this op family on XPU
#   (same recipe as the sibling bitwise_right_shift_ implementation).
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


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_kernel(a, b):
    return a << b


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_small)
@triton.jit
def bitwise_left_shift_small(a, b):
    return a << b


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_large)
@triton.jit
def bitwise_left_shift_large(a, b):
    return a << b


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_small,
)
@triton.jit
def bitwise_left_shift_scalar_small(a, b):
    return a << b


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_large,
)
@triton.jit
def bitwise_left_shift_scalar_large(a, b):
    return a << b


def bitwise_left_shift(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN BITWISE_LEFT_SHIFT")
    if out is None:
        return bitwise_left_shift_kernel(self, other)
    return bitwise_left_shift_kernel(self, other, out=out)


def bitwise_left_shift_(self, other):
    logger.debug("GEMS_KUNLUNXIN BITWISE_LEFT_SHIFT_")
    large = self.numel() > 4096
    if isinstance(other, torch.Tensor):
        if large:
            return bitwise_left_shift_large(self, other, out0=self)
        return bitwise_left_shift_small(self, other, out0=self)
    if large:
        return bitwise_left_shift_scalar_large(self, other, out0=self)
    return bitwise_left_shift_scalar_small(self, other, out0=self)
