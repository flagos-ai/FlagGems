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

_SMALL_NUMEL = 4096

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


def _left_shift_kernel(a, b, *, out=None, out0=None):
    large = a.numel() > _SMALL_NUMEL
    if out is not None:
        if large:
            return bitwise_left_shift_large(a, b, out0=out)
        return bitwise_left_shift_small(a, b, out0=out)
    if out0 is not None:
        if large:
            return bitwise_left_shift_large(a, b, out0=out0)
        return bitwise_left_shift_small(a, b, out0=out0)
    if large:
        return bitwise_left_shift_large(a, b)
    return bitwise_left_shift_small(a, b)


def bitwise_left_shift(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN BITWISE_LEFT_SHIFT")
    if not isinstance(other, torch.Tensor):
        # Tensor_Scalar entry: keep the generic kernel (pre-change behavior).
        return bitwise_left_shift_kernel(self, other)
    if (
        self.dtype == other.dtype
        and self.shape == other.shape
        and self.is_contiguous()
        and other.is_contiguous()
    ):
        # Same-shape/same-dtype/contiguous fast path: allocate with
        # torch.empty_strided (allocation only, no compute) and pass out0= so
        # the generated wrapper skips per-call promotion/allocation (same
        # recipe as bitwise_and_tensor).
        out = torch.empty_strided(
            self.shape, self.stride(), dtype=self.dtype, device=self.device
        )
        return _left_shift_kernel(self, other, out0=out)
    return _left_shift_kernel(self, other, out=out)


def bitwise_left_shift_(self, other):
    logger.debug("GEMS_KUNLUNXIN BITWISE_LEFT_SHIFT_")
    if isinstance(other, torch.Tensor):
        return _left_shift_kernel(self, other, out0=self)
    if self.numel() > _SMALL_NUMEL:
        return bitwise_left_shift_scalar_large(self, other, out0=self)
    return bitwise_left_shift_scalar_small(self, other, out0=self)
