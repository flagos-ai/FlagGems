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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("adaptive_max_pool3d_backward"),
    key=["n_input_elements"],
)
@triton.jit
def adaptive_max_pool3d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    in_stride_c,
    out_stride_c,
    # Maximum number of output positions that can overlap one input position
    # in each dimension.  Computed as ceil(in_size / out_size) + 1.
    MAX_OD: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
    n_input_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Input-centric backward with bounded window iteration.

    For each input position we compute the tight range of output positions
    whose adaptive-pooling window covers it, then iterate only over that
    small range (typically 2-3 per dimension).  No atomic operations.
    """
    pid = tl.program_id(0)
    nc_idx = tl.program_id(1)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_spatial = in_d * in_h * in_w
    mask = offsets < in_spatial

    hw = in_h * in_w
    d_in = offsets // hw
    rem = offsets % hw
    h_in = rem // in_w
    w_in = rem % in_w

    my_flat_idx = offsets

    grad_out_base = grad_output_ptr + nc_idx * out_stride_c
    indices_base = indices_ptr + nc_idx * out_stride_c
    grad_in_base = grad_input_ptr + nc_idx * in_stride_c

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    out_hw = out_h * out_w

    # For adaptive pooling:
    #   start(o) = floor(o * in_size / out_size)
    #   end(o)   = ceil((o+1) * in_size / out_size)
    # Given input position p, output o covers p iff start(o) <= p < end(o).
    # Tight output range for p:
    #   o_lo = max(0, ceil((p+1) * out_size / in_size) - MAX_span)
    #        but simplest correct: o_lo that satisfies end(o_lo) > p
    #   We use: o_lo = max(0, (p * out_size + in_size - 1) // in_size - 1)
    #           o_hi = min(out_size, (p + 1) * out_size // in_size + 1)
    # Then iterate o from o_lo to o_lo + MAX_O (constexpr bound), mask by < o_hi.

    # D dimension bounds
    o_d_lo = tl.maximum((d_in * out_d) // in_d, 0)
    o_d_hi = tl.minimum(((d_in + 1) * out_d + in_d - 1) // in_d, out_d)

    # H dimension bounds
    o_h_lo = tl.maximum((h_in * out_h) // in_h, 0)
    o_h_hi = tl.minimum(((h_in + 1) * out_h + in_h - 1) // in_h, out_h)

    # W dimension bounds
    o_w_lo = tl.maximum((w_in * out_w) // in_w, 0)
    o_w_hi = tl.minimum(((w_in + 1) * out_w + in_w - 1) // in_w, out_w)

    # Iterate over bounded window using constexpr loop bounds
    for delta_d in range(MAX_OD):
        o_d = o_d_lo + delta_d
        d_valid = mask & (o_d < o_d_hi)

        for delta_h in range(MAX_OH):
            o_h = o_h_lo + delta_h
            h_valid = d_valid & (o_h < o_h_hi)

            for delta_w in range(MAX_OW):
                o_w = o_w_lo + delta_w
                w_valid = h_valid & (o_w < o_w_hi)

                out_offset = o_d * out_hw + o_h * out_w + o_w

                idx_val = tl.load(
                    indices_base + out_offset,
                    mask=w_valid,
                    other=-1,
                )

                match = w_valid & (idx_val == my_flat_idx)

                grad_val = tl.load(
                    grad_out_base + out_offset,
                    mask=match,
                    other=0.0,
                )

                acc += tl.where(match, grad_val, 0.0)

    tl.store(
        grad_in_base + offsets,
        acc.to(grad_input_ptr.type.element_ty),
        mask=mask,
    )


def adaptive_max_pool3d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    indices: torch.Tensor,
):
    logger.debug("GEMS_HYGON ADAPTIVE_MAX_POOL3D_BACKWARD")

    in_n, in_c, in_d, in_h, in_w = self.shape
    out_d, out_h, out_w = (
        grad_output.shape[2],
        grad_output.shape[3],
        grad_output.shape[4],
    )

    grad_input = torch.empty_like(self)

    if grad_input.numel() == 0:
        return grad_input

    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    nc_count = in_n * in_c
    in_spatial = in_d * in_h * in_w
    out_spatial = out_d * out_h * out_w

    # Maximum overlap per dimension: ceil(in / out) + 1
    # This is the max number of output positions whose window can cover
    # a single input position.
    max_od = math.ceil(in_d / out_d) + 1 if out_d > 0 else 1
    max_oh = math.ceil(in_h / out_h) + 1 if out_h > 0 else 1
    max_ow = math.ceil(in_w / out_w) + 1 if out_w > 0 else 1

    grid = lambda meta: (
        triton.cdiv(in_spatial, meta["BLOCK_SIZE"]),
        nc_count,
    )

    with torch_device_fn.device(self.device):
        adaptive_max_pool3d_backward_kernel[grid](
            grad_output,
            indices,
            grad_input,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            in_spatial,
            out_spatial,
            max_od,
            max_oh,
            max_ow,
            in_spatial,
        )

    return grad_input
