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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_stages=2, num_warps=4),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
)
@triton.jit
def _adaptive_avg_pool2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    BLOCK: tl.constexpr,
):
    """Compute one linear input tile by accumulating its covering windows."""
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    linear_offsets = pid_hw * BLOCK + tl.arange(0, BLOCK)
    h_pos = linear_offsets // in_w
    w_pos = linear_offsets % in_w
    input_mask = linear_offsets < in_h * in_w

    grad_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    grad_output_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c

    # Each input element belongs to one or more adaptive output windows.  The
    # input-centric mapping gives every element a single writer and needs no atomics.
    for oh in range(out_h):
        start_h = (oh * in_h) // out_h
        end_h = ((oh + 1) * in_h + out_h - 1) // out_h
        h_window = (h_pos >= start_h) & (h_pos < end_h)

        for ow in range(out_w):
            start_w = (ow * in_w) // out_w
            end_w = ((ow + 1) * in_w + out_w - 1) // out_w
            w_window = (w_pos >= start_w) & (w_pos < end_w)
            window_mask = input_mask & h_window & w_window

            window_size = (end_h - start_h) * (end_w - start_w)
            grad_out_ptr = grad_output_base + oh * out_stride_h + ow * out_stride_w
            grad_out = tl.load(grad_out_ptr).to(tl.float32)
            grad_acc += tl.where(window_mask, grad_out / window_size, 0.0)

    grad_input_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    grad_input_ptrs = grad_input_base + h_pos * in_stride_h + w_pos * in_stride_w
    tl.store(
        grad_input_ptrs,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=input_mask,
    )


def _adaptive_avg_pool2d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _ADAPTIVE_AVG_POOL2D_BACKWARD")

    input_is_3d = self.dim() == 3
    if input_is_3d:
        grad_output = grad_output.unsqueeze(0)
        self = self.unsqueeze(0)

    in_n, in_c, in_h, in_w = self.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]
    grad_input = torch.empty_like(self, dtype=self.dtype)

    if grad_output.numel() == 0 or self.numel() == 0:
        return grad_input.squeeze(0) if input_is_3d else grad_input

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_h * in_w, meta["BLOCK"]),
    )
    _adaptive_avg_pool2d_backward_kernel[grid](
        grad_output,
        grad_input,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
    )

    return grad_input.squeeze(0) if input_is_3d else grad_input
