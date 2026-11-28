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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["value_scalar"])
def fill_scalar_kernel(
    out_ptr,
    N,
    value_scalar,
    BLOCK_SIZE: tl.constexpr,
    SUBBLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    pid_offset = pid * BLOCK_SIZE
    cols = tl.arange(0, SUBBLOCK_SIZE)
    num_loop = triton.cdiv(BLOCK_SIZE, SUBBLOCK_SIZE)
    for iloop in tl.range(num_loop):
        offset = pid_offset + iloop * SUBBLOCK_SIZE + cols
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


@libentry()
@triton.jit
def fill_tensor_kernel(
    out_ptr,
    N,
    value_ptr,
    BLOCK_SIZE: tl.constexpr,
    SUBBLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    pid_offset = pid * BLOCK_SIZE
    cols = tl.arange(0, SUBBLOCK_SIZE)
    num_loop = triton.cdiv(BLOCK_SIZE, SUBBLOCK_SIZE)
    for iloop in tl.range(num_loop):
        offset = pid_offset + iloop * SUBBLOCK_SIZE + cols
        value_scalar = tl.load(value_ptr)  # load the value from the tensor.
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


def fill_tensor(input, value):
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if torch.is_grad_enabled() and (input.requires_grad or value.requires_grad):
        # The flat kernels below are not differentiable; use the generic
        # pointwise fill, which wraps the store in an autograd.Function.
        from flag_gems.ops.fill import fill_tensor as _generic_fill_tensor

        return _generic_fill_tensor(input, value)
    logger.debug("GEMS_ASCEND FILL")
    if value.device != input.device:
        # CPU scalar tensor: fetch it once instead of staging a device load.
        return fill_scalar(input, value.item())
    out = torch.empty_like(input)
    N = out.numel()
    if N == 0:
        return out
    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid,](out, N, value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return out


def fill_scalar(input, value):
    if torch.is_grad_enabled() and input.requires_grad:
        from flag_gems.ops.fill import fill_scalar as _generic_fill_scalar

        return _generic_fill_scalar(input, value)
    logger.debug("GEMS_ASCEND FILL")
    out = torch.empty_like(input)
    N = out.numel()
    if N == 0:
        return out
    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid,](out, N, value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return out


def fill_tensor_(self, value):
    if torch.is_grad_enabled() and (self.requires_grad or value.requires_grad):
        from flag_gems.ops.fill import fill_tensor_ as _generic_fill_tensor_

        return _generic_fill_tensor_(self, value)
    if not self.is_contiguous():
        # The kernels below store to flat offsets 0..numel-1 from the view's
        # data pointer and ignore strides, so a strided view would be written
        # outside its own memory while part of the view stays untouched. The
        # generic pointwise fill walks the view's strides instead.
        from flag_gems.ops.fill import fill_tensor_ as _strided_fill_tensor_

        return _strided_fill_tensor_(self, value)
    logger.debug("GEMS_ASCEND FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    N = self.numel()
    if N == 0:
        return self
    if value.device != self.device:
        return fill_scalar_(self, value.item())
    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(self.device):
        fill_tensor_kernel[grid,](self, N, value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return self


def fill_scalar_(self, value):
    logger.debug("GEMS_ASCEND FILL_SCALAR_")
    N = self.numel()
    if N == 0:
        return self
    if torch.is_grad_enabled() and self.requires_grad:
        from flag_gems.ops.fill import fill_scalar_ as _generic_fill_scalar_

        return _generic_fill_scalar_(self, value)
    if not self.is_contiguous():
        # Same stride hazard as `fill_tensor_`: the flat store kernel would
        # write outside a strided view. Use the stride-aware generic fill.
        from flag_gems.ops.fill import fill_scalar_ as _strided_fill_scalar_

        return _strided_fill_scalar_(self, value)
    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(self.device):
        fill_scalar_kernel[grid,](self, N, value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return self
