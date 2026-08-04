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

try:
    from triton.language.extra.cuda.libdevice import pow as _pow
except ImportError:
    try:
        from triton.language.math import pow as _pow
    except ImportError:
        from triton.language.libdevice import pow as _pow


logger = logging.getLogger(__name__)


def _is_float64_scalar(*args):
    return any(
        isinstance(a, torch.Tensor) and a.dtype == torch.float64 and a.ndim == 0
        for a in args
    )


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_ENFLAME POW_TENSOR_TENSOR")
    if _is_float64_scalar(A, exponent):
        device = A.device
        return torch.pow(A.cpu(), exponent.cpu()).to(device)
    if exponent.dtype == torch.int64:
        exponent = exponent.to(torch.int32)
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_ENFLAME POW_TENSOR_TENSOR_")
    if _is_float64_scalar(A, exponent):
        A.copy_(torch.pow(A.cpu(), exponent.cpu()))
        return A
    if exponent.dtype == torch.int64:
        exponent = exponent.to(torch.int32)
    if A.dtype == torch.int64:
        A = A.to(torch.int32)
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_ENFLAME POW_TENSOR_SCALAR")
    if _is_float64_scalar(A):
        return torch.pow(A.cpu(), exponent).to(A.device)
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_ENFLAME POW_TENSOR_SCALAR_")
    if _is_float64_scalar(A):
        A.copy_(torch.pow(A.cpu(), exponent))
        return A
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    logger.debug("GEMS_ENFLAME POW_SCALAR")
    if _is_float64_scalar(exponent):
        return torch.pow(A, exponent.cpu()).to(exponent.device)
    return pow_func_scalar_tensor(A, exponent)
