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

logger = logging.getLogger(__name__)


@triton.jit
def logit_backward_kernel(
    grad_out_ptr,
    self_ptr,
    y_ptr,
    n_elements,
    lo,
    hi,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    self_vals = tl.load(self_ptr + offsets, mask=mask, other=0)
    self_f32 = self_vals.to(tl.float32)

    self_clamped = tl.minimum(tl.maximum(self_f32, lo), hi)
    in_range = (self_f32 >= lo) & (self_f32 <= hi)
    grad_input = tl.where(
        in_range,
        grad_out.to(tl.float32) / (self_clamped * (1.0 - self_clamped)),
        0.0,
    )
    tl.store(y_ptr + offsets, grad_input.to(OUT_DTYPE), mask=mask)


def _to_triton_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return None


def _logit_backward_impl(grad_output: torch.Tensor, self: torch.Tensor, eps=None):
    if not isinstance(grad_output, torch.Tensor):
        raise TypeError("grad_output must be a torch.Tensor")
    if not isinstance(self, torch.Tensor):
        raise TypeError("self must be a torch.Tensor")
    if not self.is_floating_point():
        raise TypeError("logit_backward expected self to be a floating point tensor")

    if eps is not None:
        eps = float(eps)
        if not (0.0 <= eps <= 0.5):
            raise ValueError("eps must be in the range [0.0, 0.5].")
        lo = eps
        hi = 1.0 - eps
    else:
        lo = 0.0
        hi = 1.0

    grad_contig = grad_output.contiguous()
    self_contig = self.contiguous()

    in_supported = _to_triton_dtype(grad_contig.dtype) is not None
    grad_kernel = grad_contig if in_supported else grad_contig.to(torch.float32)
    self_kernel = self_contig if in_supported else self_contig.to(torch.float32)

    desired_dtype = self.dtype
    desired_supported = _to_triton_dtype(desired_dtype) is not None
    if desired_supported:
        result = torch.empty_like(self, dtype=desired_dtype)
        work_out = result
    else:
        work_out = torch.empty_like(self, dtype=torch.float32)

    n_elements = grad_kernel.numel()
    # A fixed 1024-element tile matches the pointwise kernel's one-value-per-element work pattern.
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    triton_dtype = _to_triton_dtype(work_out.dtype)
    with torch_device_fn.device(self.device):
        logit_backward_kernel[grid](
            grad_kernel,
            self_kernel,
            work_out,
            n_elements,
            lo,
            hi,
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DTYPE=triton_dtype,
        )

    if desired_supported:
        return work_out
    else:
        return work_out.to(desired_dtype)


def logit_backward(grad_output, self, eps=None):
    logger.debug("GEMS_KUNLUNXIN LOGIT_BACKWARD")
    return _logit_backward_impl(grad_output, self, eps=eps)
