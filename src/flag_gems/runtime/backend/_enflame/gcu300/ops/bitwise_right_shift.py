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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_func(x, y):
    return x >> y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_func_scalar(x, y):
    return x >> y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_scalar_func(x, y):
    return x >> y


def _uint8_operands(*args):
    """Return ``(args, needs_narrowing)`` with uint8 operands widened to int32.

    GCU300 cannot link the uint8 shift lowering (``ld.lld: relocation ...
    acore_shrui_ui8``), so 8-bit shifts are computed in int32 and the result is
    narrowed back by the caller.
    """
    widened = []
    needs_narrowing = False
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.dtype == torch.uint8:
            widened.append(arg.to(torch.int32))
            needs_narrowing = True
        else:
            widened.append(arg)
    return widened, needs_narrowing


def bitwise_right_shift(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_RIGHT_SHIFT")
    if isinstance(A, torch.Tensor) or isinstance(B, torch.Tensor):
        (A, B), narrow = _uint8_operands(A, B)
        if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
            out = bitwise_right_shift_func(A, B)
        elif isinstance(A, torch.Tensor):
            out = bitwise_right_shift_func_scalar(A, B)
        else:
            # `A` is the scalar and `B` the tensor, so the shift is A >> B; the
            # tensor has to be the kernel's first argument.
            out = bitwise_right_shift_scalar_func(A, B)
        return out.to(torch.uint8) if narrow else out
    # Both scalar
    return torch.tensor(A >> B)


def bitwise_right_shift_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_RIGHT_SHIFT_")
    (A32, B32), narrow = _uint8_operands(A, B)
    if not narrow:
        if isinstance(B, torch.Tensor):
            return bitwise_right_shift_func(A, B, out0=A)
        return bitwise_right_shift_func_scalar(A, B, out0=A)
    # The in-place target keeps its uint8 dtype, so the int32 result is narrowed
    # into it instead of being written directly.
    if isinstance(B32, torch.Tensor):
        out = bitwise_right_shift_func(A32, B32)
    else:
        out = bitwise_right_shift_func_scalar(A32, B32)
    return A.copy_(out.to(torch.uint8))
