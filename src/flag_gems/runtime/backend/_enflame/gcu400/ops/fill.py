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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_INT32_MAX = torch.iinfo(torch.int32).max
# Element-wise Triton stores of 64-bit dtypes are not reliable on this backend
# (the runtime replaces Long/Double with Int, corrupting every other element),
# so they stay on the generic broadcast kernel.
_FLAT_UNSAFE_DTYPES = (torch.int64, torch.uint64, torch.float64)
_FLAT_MAX_BLOCK = 65536
_FLAT_MAX_GRID = 48


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


@libentry()
@triton.jit
def fill_tensor_flat_kernel(
    out_ptr,
    value_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # The value is a 0-dim device tensor: load it once per CTA and broadcast
    # it on the store, instead of materializing a stride-0 block load per tile.
    fill_value = tl.load(value_ptr)
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        if EXACT:
            tl.store(out_ptr + offsets, fill_value)
        else:
            mask = offsets < n_elements
            tl.store(out_ptr + offsets, fill_value, mask=mask)


def _fill_tensor_flat_eligible(input, out, value):
    if out.dtype in _FLAT_UNSAFE_DTYPES or value.dtype in _FLAT_UNSAFE_DTYPES:
        return False
    if value.dtype != out.dtype:
        return False
    if out.numel() > _INT32_MAX:
        return False
    if out.shape != input.shape:
        return False
    # A uniform fill writes the same value to every element, so any dense
    # non-overlapping layout can be filled in flat memory order.
    return torch.ops.aten.is_non_overlapping_and_dense(out)


def _fill_tensor_flat(out, value):
    n_elements = out.numel()
    if n_elements == 0:
        return
    block = min(_FLAT_MAX_BLOCK, triton.next_power_of_2(n_elements))
    grid = (min(_FLAT_MAX_GRID, triton.cdiv(n_elements, block)),)
    fill_tensor_flat_kernel[grid](
        out,
        value,
        n_elements,
        BLOCK=block,
        EXACT=(n_elements % block == 0),
        num_warps=4,
    )


def fill_scalar(input, value):
    logger.debug("GEMS_ENFLAME FILL")
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS_ENFLAME FILL")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    if _fill_tensor_flat_eligible(input, out, value):
        with torch_device_fn.device(input.device):
            _fill_tensor_flat(out, value)
        return out
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS_ENFLAME FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS_ENFLAME FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if _fill_tensor_flat_eligible(input, out, value):
        with torch_device_fn.device(input.device):
            _fill_tensor_flat(out, value)
        return out
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=out)
    return out


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS_ENFLAME FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if _fill_tensor_flat_eligible(self, self, value):
        with torch_device_fn.device(self.device):
            _fill_tensor_flat(self, value)
        return self
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self


def fill_scalar_(self, value=0):
    logger.debug("GEMS_ENFLAME FILL_SCALAR_")
    with torch_device_fn.device(self.device):
        fill_scalar_func(self, value, out0=self)
    return self
