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

from flag_gems.runtime import device
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


@triton.jit
def _maximum_values(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)
    return tl.maximum(X, Y, propagate_nan=tl.PropagateNan.ALL)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def maximum_kernel(X, Y):
    return _maximum_values(X, Y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def maximum_kernel_tensor_scalar(X, Y):
    return _maximum_values(X, Y.to(X.dtype))


def _check_backend(tensor):
    if tensor.device.type != device:
        raise RuntimeError(
            f"maximum(): expected a tensor on the {device} backend, but got "
            f"{tensor.device}"
        )


def _is_cpu_0d_tensor(tensor):
    return tensor.dim() == 0 and tensor.device.type == "cpu"


def _host_scalar(scalar, result_dtype):
    # Keep integral scalars as Python ints so int64/int32 values reach the
    # kernel without a detour through fp32.
    value = scalar.item()
    if result_dtype is torch.bool:
        return bool(value)
    if not scalar.dtype.is_floating_point:
        return int(value)
    return float(value)


def _maximum_with_cpu_scalar(scalar, tensor):
    _check_backend(tensor)
    # ATen treats a 0-dim operand as a wrapped scalar, so promotion must use
    # the original tensor, not the Python scalar.
    result_dtype = torch.result_type(scalar, tensor)
    # The scalar kernel promotes against a plain Python value, which no longer
    # carries the wrapped scalar's dtype; align the tensor with the already
    # resolved result dtype first.
    if tensor.dtype != result_dtype:
        tensor = tensor.to(result_dtype)
    # Python floats reach Triton as fp32. Materialise an fp64 scalar on the
    # device so its low mantissa bits survive.
    if scalar.dtype == torch.float64:
        return maximum_kernel(tensor, scalar.to(tensor))
    return maximum_kernel_tensor_scalar(tensor, _host_scalar(scalar, result_dtype))


def maximum(X, Y):
    logger.debug("GEMS MAXIMUM")
    if X.device == Y.device:
        _check_backend(X)
        return maximum_kernel(X, Y)
    if _is_cpu_0d_tensor(X):
        return _maximum_with_cpu_scalar(X, Y)
    if _is_cpu_0d_tensor(Y):
        return _maximum_with_cpu_scalar(Y, X)
    raise RuntimeError(
        "Expected all tensors to be on the same device, but found at least "
        f"two devices, {X.device} and {Y.device}!"
    )
