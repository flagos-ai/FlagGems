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

from flag_gems.ops.neg import neg as _generic_neg
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024
_INT32_MAX = 2**31 - 1
_DEVICE_TYPES = ("cuda", torch._C._get_privateuse1_backend_name())


@triton.jit
def _neg_contiguous_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # IEEE-754 negation changes only the sign bit, including signed zero,
    # subnormals, infinities and NaN payloads. No floating point arithmetic.
    bits = x.to(tl.uint32, bitcast=True) ^ tl.full((), 0x80000000, tl.uint32)
    tl.store(output_ptr + offsets, bits.to(tl.float32, bitcast=True), mask=mask)


def _can_use_fast_path(A):
    # Metadata only. Leave strides, dtype promotion, lazy views, subclasses and
    # autograd inputs to the existing generic implementation.
    return (
        type(A) is torch.Tensor
        and A.layout == torch.strided
        and A.device.type in _DEVICE_TYPES
        and A.dtype == torch.float32
        and not A.requires_grad
        and not A.is_conj()
        and not A.is_neg()
        and A.is_contiguous()
        and A.numel() <= _INT32_MAX
    )


def neg(A):
    logger.debug("GEMS_HYGON NEG")
    if not _can_use_fast_path(A):
        return _generic_neg(A)
    with torch_device_fn.device(A.device.index):
        output = torch.empty_like(A)
        n_elements = A.numel()
        if n_elements:
            _neg_contiguous_kernel[(triton.cdiv(n_elements, _BLOCK_SIZE),)](
                A, output, n_elements, BLOCK_SIZE=_BLOCK_SIZE, num_warps=4
            )
    return output
