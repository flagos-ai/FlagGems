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

SHIFT_PROMOTE = {torch.uint8, torch.int8}

@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_func(x, y):
    return x >> y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_func_scalar(x, y):
    return x >> y


def _promote_shift_args(A, B):
    """Promote uint8/int8 operands so the kernel avoids acore_shrui_ui8."""
    out_dtype = None
    if isinstance(A, torch.Tensor) and A.dtype in SHIFT_PROMOTE:
        out_dtype = A.dtype
        A = A.to(torch.int16)
        if isinstance(B, torch.Tensor) and B.dtype in SHIFT_PROMOTE:
            B = B.to(torch.int16)
    elif isinstance(B, torch.Tensor) and B.dtype in SHIFT_PROMOTE:
        # Scalar_Tensor overload: result dtype follows B
        out_dtype = B.dtype
        B = B.to(torch.int16)
        if isinstance(A, torch.Tensor) and A.dtype in SHIFT_PROMOTE:
            A = A.to(torch.int16)
    return A, B, out_dtype


def bitwise_right_shift(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_RIGHT_SHIFT")
    A, B, out_dtype = _promote_shift_args(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        out = bitwise_right_shift_func(A, B)
    elif isinstance(A, torch.Tensor):
        out = bitwise_right_shift_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        out = bitwise_right_shift_func_scalar(B, A)
    else:
        return torch.tensor(A >> B)
    if out_dtype is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def bitwise_right_shift_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_RIGHT_SHIFT_")
    orig_dtype = A.dtype
    A_p, B_p, _ = _promote_shift_args(A, B)
    if A_p is not A:
        # In-place on promoted buffer then copy back
        if isinstance(B_p, torch.Tensor):
            tmp = bitwise_right_shift_func(A_p, B_p)
        else:
            tmp = bitwise_right_shift_func_scalar(A_p, B_p)
        A.copy_(tmp.to(orig_dtype))
        return A
    if isinstance(B, torch.Tensor):
        return bitwise_right_shift_func(A, B, out0=A)
    return bitwise_right_shift_func_scalar(A, B, out0=A)
