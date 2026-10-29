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
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic
from .bitwise_or import bitwise_or_scalar, config_

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def or_bool_int_scalar_func(x, y):
    return x.to(tl.int64) | y.to(tl.int64)


def or_scalar(self, other):
    logger.debug("GEMS_KUNLUNXIN OR_SCALAR")
    if (
        self.dtype == torch.bool
        and isinstance(other, int)
        and not isinstance(other, bool)
        and -(1 << 31) <= int(other) < (1 << 31)
    ):
        return or_bool_int_scalar_func(self, other)
    if (
        self.dtype == torch.bool
        and isinstance(other, int)
        and not isinstance(other, bool)
    ):
        # Out-of-int32-range scalar: keep the generic promotion semantics
        # (int64 result) instead of truncating through the scalar kernel.
        from flag_gems.ops.or_scalar import or_scalar as _generic_or_scalar

        return _generic_or_scalar(self, other)
    return bitwise_or_scalar(self, other)
