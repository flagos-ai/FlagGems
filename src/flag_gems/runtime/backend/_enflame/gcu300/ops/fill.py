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

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_tensor_func(inp, value):
    return value


def fill_scalar(input, value):
    logger.debug("GEMS_ENFLAME FILL")
    return_type = input.dtype
    if return_type == torch.int64:
        input = input.to(torch.int32)
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out).to(return_type)


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS_ENFLAME FILL")
    return_type = input.dtype
    if return_type == torch.int64:
        input = input.to(torch.int32)
    if value.dtype == torch.int64:
        value = value.to(torch.int32)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out).to(return_type)


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS_ENFLAME FILL_TENSOR_")
    return_type = self.dtype
    if return_type == torch.int64:
        self = self.to(torch.int32)
    if value.dtype == torch.int64:
        value = value.to(torch.int32)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self.to(return_type)


def fill_scalar_(self, value=0):
    logging.debug("GEMS_ENFLAME FILL_SCALAR_")
    return_type = self.dtype
    if return_type == torch.int64:
        self = self.to(torch.int32)
    with torch_device_fn.device(self.device):
        fill_scalar_func(self, value, out0=self)
    return self.to(return_type)


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS_ENFLAME FILL_SCALAR_")
    if out is None:
        return fill_scalar(input, value)
    return_type = input.dtype
    if return_type == torch.int64:
        input = input.to(torch.int32)
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def fill_tensor_out(input, value, *, out=None):
    if out is None:
        return fill_tensor(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    return_type = input.dtype
    if return_type == torch.int64:
        input = input.to(torch.int32)
    if value.dtype == torch.int64:
        value = value.to(torch.int32)
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=out)
    return out
