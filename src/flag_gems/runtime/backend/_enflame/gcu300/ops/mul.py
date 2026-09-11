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


def _to_compute_dtype(result_dtype):
    if result_dtype == torch.int64:
        return torch.int32
    return result_dtype if result_dtype is not None else torch.int32


def _to_compute_tensor(x, result_dtype=None):
    if not isinstance(x, torch.Tensor):
        return x
    compute_dtype = _to_compute_dtype(result_dtype)
    if x.dtype == torch.bool:
        return x.to(compute_dtype)
    if x.dtype == torch.int64:
        return x.to(torch.int32)
    return x


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func_scalar(x, y):
    return x * y


def mul(A, B):
    logger.debug("GEMS_ENFLAME MUL")
    if _is_float64_scalar(A, B):
        device = A.device if isinstance(A, torch.Tensor) else B.device
        A_cpu = A.cpu() if isinstance(A, torch.Tensor) else A
        B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
        return torch.mul(A_cpu, B_cpu).to(device)
    out_dtype = torch.result_type(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        A = _to_compute_tensor(A, out_dtype)
        B = _to_compute_tensor(B, out_dtype)
        return mul_func(A, B).to(out_dtype)
    elif isinstance(A, torch.Tensor):
        A = _to_compute_tensor(A, out_dtype)
        return mul_func_scalar(A, B).to(out_dtype)
    elif isinstance(B, torch.Tensor):
        B = _to_compute_tensor(B, out_dtype)
        return mul_func_scalar(B, A).to(out_dtype)
    else:
        # Both scalar
        return torch.tensor(A * B, dtype=out_dtype)


def mul_(A, B):
    logger.debug("GEMS_ENFLAME MUL_")
    if _is_float64_scalar(A, B):
        A_cpu = A.cpu()
        B_cpu = B.cpu() if isinstance(B, torch.Tensor) else B
        A.copy_(torch.mul(A_cpu, B_cpu))
        return A
    out_dtype = A.dtype
    compute_dtype = torch.result_type(A, B)
    if out_dtype == torch.int64:
        A_compute = A.to(torch.int32)
        if isinstance(B, torch.Tensor):
            B_compute = _to_compute_tensor(B, compute_dtype)
            result = mul_func(A_compute, B_compute)
        else:
            result = mul_func_scalar(A_compute, B)
        A.copy_(result.to(out_dtype))
        return A
    if isinstance(B, torch.Tensor):
        B = _to_compute_tensor(B, compute_dtype)
        if A.dtype == torch.bool:
            result = mul_func(_to_compute_tensor(A, compute_dtype), B)
            A.copy_(result.to(out_dtype))
            return A
        return mul_func(A, B, out0=A)
    else:
        if A.dtype == torch.bool:
            result = mul_func_scalar(_to_compute_tensor(A, compute_dtype), B)
            A.copy_(result.to(out_dtype))
            return A
        return mul_func_scalar(A, B, out0=A)
