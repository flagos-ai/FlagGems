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
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool1d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        # PyTorch-compatible adjustment for ceil_mode: the last window must
        # start strictly inside the (padded) input, not in the padding.
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 2048}, num_warps=8),
    ],
    key=["out_l", "total", "kernel_size", "stride"],
)
@triton.jit
def max_pool1d_forward_kernel(
    input_ptr,
    output_ptr,
    in_l,
    out_l,
    total,  # N * C * out_l
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each program handles a contiguous tile of the flattened output space
    # (N * C * out_l). Flattening keeps every lane busy even when out_l is
    # small, which maximizes occupancy across plane counts.
    pid = tl.program_id(0)
    out_idx = pid * BLOCK + tl.arange(0, BLOCK)
    out_mask = out_idx < total

    nc_idx = out_idx // out_l
    ol_idx = out_idx % out_l

    in_base = nc_idx * in_l
    # Window start (in the padded coordinate system) for each output element.
    start = ol_idx * stride - padding

    max_val = tl.full((BLOCK,), get_dtype_min(input_ptr.dtype.element_ty), tl.float32)

    for k in tl.static_range(0, kernel_size):
        pos = start + k * dilation
        valid = out_mask & (pos >= 0) & (pos < in_l)
        safe_pos = tl.where(valid, pos, 0)
        val = tl.load(input_ptr + in_base + safe_pos, mask=valid, other=float("-inf"))
        val = val.to(tl.float32)
        max_val = tl.maximum(max_val, val)

    tl.store(
        output_ptr + out_idx, max_val.to(output_ptr.dtype.element_ty), mask=out_mask
    )


def _parse_1d_param(param, name, default=None):
    if param is None or (isinstance(param, (list, tuple)) and len(param) == 0):
        return default
    if isinstance(param, int):
        return param
    if isinstance(param, (list, tuple)) and len(param) == 1:
        return param[0]
    raise ValueError(f"Invalid {name}: {param}")


def max_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Max pooling operation over 1D input, implemented with a Triton kernel."""
    logger.debug("GEMS MAX_POOL1D")

    assert input.ndim in (2, 3), f"max_pool1d expects 2D or 3D input, got {input.ndim}D"

    kernel_w = _parse_1d_param(kernel_size, "kernel_size")
    # stride defaults to kernel_size (PyTorch behavior).
    stride_w = _parse_1d_param(stride, "stride", default=kernel_w)
    padding_w = _parse_1d_param(padding, "padding", default=0)
    dilation_w = _parse_1d_param(dilation, "dilation", default=1)

    if stride_w <= 0:
        raise ValueError(f"stride must be positive, but got stride={stride_w}")
    if padding_w < 0:
        raise ValueError(f"padding must be non-negative, but got padding={padding_w}")
    if dilation_w <= 0:
        raise ValueError(f"dilation must be positive, but got dilation={dilation_w}")

    input = input.contiguous()

    unbatched = input.ndim == 2
    x = input.unsqueeze(0) if unbatched else input  # (N, C, L)
    in_n, in_c, in_l = x.shape

    out_l = max_pool1d_output_size(
        in_l, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty((in_n, in_c, out_l), device=input.device, dtype=input.dtype)

    if output.numel() == 0:
        return output.squeeze(0) if unbatched else output

    total = in_n * in_c * out_l
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK"]),)

    max_pool1d_forward_kernel[grid](
        x,
        output,
        in_l,
        out_l,
        total,
        kernel_w,
        stride_w,
        padding_w,
        dilation_w,
    )

    if unbatched:
        output = output.squeeze(0)
    return output
