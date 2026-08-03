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


def _is_float64_scalar(*args):
    return any(
        isinstance(a, torch.Tensor) and a.dtype == torch.float64 and a.ndim == 0
        for a in args
    )


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func_no_alpha(x, y):
    return x + y


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def add(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME ADD")
    if _is_float64_scalar(A, B):
        device = A.device if isinstance(A, torch.Tensor) else B.device
        A_cpu = A.cpu() if isinstance(A, torch.Tensor) else A
        B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
        return torch.add(A_cpu, B_cpu, alpha=alpha).to(device)
    if alpha == 1 and isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return add_func_no_alpha(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME ADD_")
    if _is_float64_scalar(A, B):
        A_cpu = A.cpu()
        B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
        A.copy_(torch.add(A_cpu, B_cpu, alpha=alpha))
        return A
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        if B.dtype == torch.int64:
            B = B.to(torch.int32)
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        if A.dtype == torch.int64:
            A = A.to(torch.int32)
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
