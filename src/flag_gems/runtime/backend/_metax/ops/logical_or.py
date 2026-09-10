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
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


# For non-bf16 dtypes, use pointwise_dynamic as normal
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_or_func(x, y):
    return x.to(tl.int1) | y.to(tl.int1)


# MetaX workaround: The MetaX compiler does not support uitofp from i1 to bf16
# (bf16 is represented as i16 in LLVM IR on MetaX). The Triton compiler's
# optimization passes fold i1->f32->bf16 into direct i1->bf16, so we cannot
# rely on intermediate casts. Instead we use tl.where to select between
# pre-constructed bf16 constants (1.0 and 0.0), which generates a 'select'
# instruction rather than 'uitofp' and is legal on MetaX.
@libentry()
@triton.jit
def logical_or_bf16_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Compute boolean OR
    cond = (x.to(tl.int1)) | (y.to(tl.int1))
    # Store as int8 (backing type for torch.bool) to avoid uitofp i1->bf16
    result = cond.to(tl.int8)
    tl.store(out_ptr + offsets, result, mask=mask)


def _logical_or_bf16(A, B):
    """Handle bf16 inputs with manual kernel to avoid uitofp i1 -> bf16."""
    A_flat = A.contiguous().view(-1)
    B_flat = B.broadcast_to(A.shape).contiguous().view(-1)
    out_flat = torch.empty(A_flat.shape, dtype=torch.bool, device=A.device)
    n_elements = A_flat.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    with torch_device_fn.device(A.device):
        logical_or_bf16_kernel[grid](A_flat, B_flat, out_flat, n_elements, BLOCK_SIZE)
    return out_flat.view(A.shape)


def logical_or(A, B):
    logger.debug("GEMS_METAX LOGICAL_OR")
    if A.dtype == torch.bfloat16:
        return _logical_or_bf16(A, B)
    return logical_or_func(A, B)


def logical_or_(A, B):
    logger.debug("GEMS_METAX LOGICAL_OR_")
    if A.dtype == torch.bfloat16:
        # Compute logical OR using dedicated bf16 kernel (returns bool)
        result = _logical_or_bf16(A, B)
        # MetaX MLIR cannot compile bool->bf16 copy kernel;
        # convert via float32 intermediate to avoid cross-dtype MLIR issue
        A.copy_(result.to(torch.float32).to(A.dtype))
        return A
    logical_or_func(A, B, out0=A)
    return A
