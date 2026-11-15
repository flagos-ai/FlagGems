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


# ATen 的 Scalar_Tensor 重载把标量作为被移位量，不能复用 tensor-scalar kernel。
@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_func_scalar_tensor(x, y):
    return x >> y


def bitwise_right_shift(A, B):
    # debug-only logger removed
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return bitwise_right_shift_func(A, B)
    elif isinstance(A, torch.Tensor):
        return bitwise_right_shift_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return bitwise_right_shift_func_scalar_tensor(A, B)
    else:
        # 双标量时按 Python 右移语义直接构造 0-D 张量。
        return torch.tensor(A >> B)


def bitwise_right_shift_(A, B):
    # debug-only logger removed
    if isinstance(B, torch.Tensor):
        return bitwise_right_shift_func(A, B, out0=A)
    else:
        return bitwise_right_shift_func_scalar(A, B, out0=A)
