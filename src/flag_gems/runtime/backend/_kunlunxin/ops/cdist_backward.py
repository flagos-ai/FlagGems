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

# XPU override for cdist_backward.
#
# The general kernel builds contrib as a 2D [BLOCK_N2, BLOCK_DIM] tile and
# reduces along axis=0. XPU triton rejects axis=0 reductions on 2D+ tensors
# ("axis must not be 0 for 2D+ shapes"). Swapping the layout to
# [BLOCK_DIM, BLOCK_N2] with axis=1 reduce compiles but TritonXPUCoreTiling
# still fails on 64x64 tiles.
#
# This override eliminates the 2D tile entirely: loop over n2 as a scalar
# runtime loop, each step loading a 1D [BLOCK_DIM] x2 vector and scalar
# grad/cdist. All ops stay 1D so XPU codegen is clean. n2 in the current
# tests is small (n1//2+1 <= 33), so the scalar loop is not a hot spot.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _cdist_backward_kernel(
    grad_ptr,
    x1_ptr,
    x2_ptr,
    cdist_ptr,
    grad_x1_ptr,
    batch_size,
    n1,
    n2,
    dim,
    p,
    BLOCK_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n1 = tl.program_id(1)
    pid_dim = tl.program_id(2)

    off_dim = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_dim = off_dim < dim
    n1_idx = pid_n1

    # x1[b, n1_idx, off_dim] : [BLOCK_DIM]
    x1_offset = pid_b * n1 * dim + n1_idx * dim + off_dim
    x1 = tl.load(x1_ptr + x1_offset, mask=mask_dim, other=0.0).to(tl.float32)

    grad_x1_acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    grad_base = pid_b * n1 * n2 + n1_idx * n2
    x2_base = pid_b * n2 * dim

    eps = 1e-12
    for j in range(0, n2):
        # scalar loads (broadcast a 1-element load)
        gj = tl.load(grad_ptr + grad_base + j).to(tl.float32)
        cj = tl.load(cdist_ptr + grad_base + j).to(tl.float32)
        # x2[b, j, off_dim] : [BLOCK_DIM]
        x2j = tl.load(
            x2_ptr + x2_base + j * dim + off_dim, mask=mask_dim, other=0.0
        ).to(tl.float32)
        diff = x1 - x2j
        grad_x1_acc += gj * diff / (cj + eps)

    store_offset = pid_b * n1 * dim + n1_idx * dim + off_dim
    tl.store(grad_x1_ptr + store_offset, grad_x1_acc, mask=mask_dim)


def _cdist_backward(grad, x1, x2, p, cdist):
    logger.debug("GEMS_KUNLUNXIN _cdist_backward")
    assert x1.device == x2.device == grad.device == cdist.device
    assert x1.shape[0] == x2.shape[0] == grad.shape[0] == cdist.shape[0]
    assert x1.shape[2] == x2.shape[2]
    assert x1.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"Unsupported dtype: {x1.dtype}"

    batch_size, n1, dim = x1.shape
    _, n2, _ = x2.shape

    grad = grad.contiguous()
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    cdist = cdist.contiguous()

    if x1.dtype in (torch.float16, torch.bfloat16):
        grad_x1_fp32 = torch.empty_like(x1, dtype=torch.float32)
    else:
        grad_x1_fp32 = torch.empty_like(x1)

    BLOCK_DIM = 64
    grid = (batch_size, n1, triton.cdiv(dim, BLOCK_DIM))

    with torch_device_fn.device(x1.device):
        _cdist_backward_kernel[grid](
            grad,
            x1,
            x2,
            cdist,
            grad_x1_fp32,
            batch_size,
            n1,
            n2,
            dim,
            float(p),
            BLOCK_DIM=BLOCK_DIM,
        )

    if x1.dtype in (torch.float16, torch.bfloat16):
        return grad_x1_fp32.to(x1.dtype)
    return grad_x1_fp32
