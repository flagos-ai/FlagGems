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

# Two codegen variants for the two latency regimes on XPU (mirrors the
# validated sibling override _kunlunxin/ops/bitwise_right_shift.py):
# - small tensors (numel <= 4096): launch-bound -> 1-CTA (kunlunAutoGrid) path
#   keeps the whole tensor in one CTA.
# - medium/large tensors: 12-CTA grid-stride path with explicit unroll 16;
#   unroll 16 is the sweet spot measured for this op family on XPU.
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
def _rshift_tensor_kernel_small(a, b):
    return a >> b


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_large)
@triton.jit
def _rshift_tensor_kernel_large(a, b):
    return a >> b


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_small
)
@triton.jit
def _rshift_scalar_kernel_small(a, b):
    return a >> b


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_large
)
@triton.jit
def _rshift_scalar_kernel_large(a, b):
    return a >> b


def _rshift_kernel(a, b, *, out=None):
    """Dispatch to the small/large codegen variants, tensor-vs-scalar shift."""
    large = a.numel() > 4096
    if torch.is_tensor(b):
        if large:
            return _rshift_tensor_kernel_large(a, b, out0=out)
        return _rshift_tensor_kernel_small(a, b, out0=out)
    if large:
        return _rshift_scalar_kernel_large(a, b, out0=out)
    return _rshift_scalar_kernel_small(a, b, out0=out)


def __rshift__(self: torch.Tensor, other, *, out=None) -> torch.Tensor:
    """Apply arithmetic or logical right shift for tensor and scalar operands."""
    logger.debug("GEMS_KUNLUNXIN __RSHIFT__")
    if out is None:
        # Non-out path: let the pointwise_dynamic wrapper allocate the output.
        large = self.numel() > 4096
        if torch.is_tensor(other):
            if large:
                return _rshift_tensor_kernel_large(self, other)
            return _rshift_tensor_kernel_small(self, other)
        if large:
            return _rshift_scalar_kernel_large(self, other)
        return _rshift_scalar_kernel_small(self, other)
    _rshift_kernel(self, other, out=out)
    return out


def __rshift___out(self: torch.Tensor, other, *, out) -> torch.Tensor:
    """Apply arithmetic or logical right shift for tensor and scalar operands (out variant)."""
    logger.debug("GEMS_KUNLUNXIN __RSHIFT__ OUT")
    return __rshift__(self, other, out=out)
