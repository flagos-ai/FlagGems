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

from flag_gems.ops.fill import _FillScalar as _GenericFillScalar
from flag_gems.ops.fill import _FillScalarInplace as _GenericFillScalarInplace
from flag_gems.ops.fill import _FillTensor as _GenericFillTensor
from flag_gems.ops.fill import _FillTensorInplace as _GenericFillTensorInplace
from flag_gems.ops.fill import fill_scalar_func, fill_tensor_func
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def fill_scalar_kernel(
    out_ptr,
    value_scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load a dummy value to infer the dtype of out_ptr
    dummy = tl.load(out_ptr + offsets, mask=mask, other=0)
    fill_val = tl.full([BLOCK_SIZE], value_scalar, dtype=dummy.dtype)
    tl.store(out_ptr + offsets, fill_val, mask=mask)


@triton.jit
def fill_tensor_kernel(
    out_ptr,
    value_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(value_ptr)
    tl.store(out_ptr + offsets, val, mask=mask)


def _as_contiguous(tensor):
    """Return tensor.contiguous() view for use with flat-offset kernels.

    For non-contiguous tensors this allocates a new buffer; callers that
    need in-place semantics must copy back afterwards.
    """
    if tensor.is_contiguous():
        return tensor, False
    return tensor.contiguous(), True


def fill_scalar_raw(input, value):
    logger.debug("GEMS_NVIDIA FILL_SCALAR")
    out = torch.empty_like(input)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid](out, value, n_elements, BLOCK_SIZE=1024)
    return out


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS_NVIDIA FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar_raw(input, value)
    out_contig, need_copy = _as_contiguous(out)
    n_elements = out_contig.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid](out_contig, value, n_elements, BLOCK_SIZE=1024)
    if need_copy:
        out.copy_(out_contig)
    return out


def fill_tensor_raw(input, value):
    if not value.is_cuda:
        return fill_scalar_raw(input, value.item())
    logger.debug("GEMS_NVIDIA FILL_TENSOR")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid](out, value, n_elements, BLOCK_SIZE=1024)
    return out


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS_NVIDIA FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor_raw(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out_contig, need_copy = _as_contiguous(out)
    n_elements = out_contig.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid](out_contig, value, n_elements, BLOCK_SIZE=1024)
    if need_copy:
        out.copy_(out_contig)
    return out


class _FillScalar(_GenericFillScalar):
    @staticmethod
    def forward(ctx, input, value):
        return fill_scalar_raw(input, value)


class _FillTensor(_GenericFillTensor):
    @staticmethod
    def forward(ctx, input, value):
        ctx.value_device = value.device
        return fill_tensor_raw(input, value)


def fill_scalar(input, value):
    if torch.is_grad_enabled() and input.requires_grad:
        return _FillScalar.apply(input, value)
    return fill_scalar_raw(input, value)


def fill_tensor(input, value):
    if value.ndim != 0:
        raise RuntimeError("fill only supports 0-dimension value tensor")
    if torch.is_grad_enabled() and (input.requires_grad or value.requires_grad):
        return _FillTensor.apply(input, value)
    return fill_tensor_raw(input, value)


def _fill_tensor_(self, value):
    if not value.is_cuda:
        return _fill_scalar_(self, value.item())
    logger.debug("GEMS_NVIDIA FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if self.is_contiguous():
        n_elements = self.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        with torch_device_fn.device(self.device):
            fill_tensor_kernel[grid](self, value, n_elements, BLOCK_SIZE=1024)
    else:
        with torch_device_fn.device(self.device):
            fill_tensor_func(self, value, out0=self)
    return self


def _fill_scalar_(self, value):
    logger.debug("GEMS_NVIDIA FILL_SCALAR_")
    if self.is_contiguous():
        n_elements = self.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        with torch_device_fn.device(self.device):
            fill_scalar_kernel[grid](self, value, n_elements, BLOCK_SIZE=1024)
    else:
        with torch_device_fn.device(self.device):
            fill_scalar_func(self, value, out0=self)
    return self


class _FillScalarInplace(_GenericFillScalarInplace):
    @staticmethod
    def forward(ctx, input, value):
        ctx.mark_dirty(input)
        return _fill_scalar_(input, value)


class _FillTensorInplace(_GenericFillTensorInplace):
    @staticmethod
    def forward(ctx, input, value):
        ctx.value_device = value.device
        ctx.mark_dirty(input)
        return _fill_tensor_(input, value)


def fill_scalar_(self, value):
    if torch.is_grad_enabled() and self.requires_grad:
        return _FillScalarInplace.apply(self, value)
    _fill_scalar_(self, value)
    torch.autograd.graph.increment_version(self)
    return self


def fill_tensor_(self, value):
    if value.ndim != 0:
        raise RuntimeError("fill_ only supports 0-dimension value tensor")
    if torch.is_grad_enabled() and (self.requires_grad or value.requires_grad):
        return _FillTensorInplace.apply(self, value)
    _fill_tensor_(self, value)
    torch.autograd.graph.increment_version(self)
    return self
