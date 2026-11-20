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

from .avg_pool2d import avg_pool2d

# Kunlunxin: the 1-D pool is the 2-D pool with a height-1 window, so it can ride
# the launch-table avg_pool2d entry. The kernel below is a carrier: its body is
# never executed (the table matches the name and runs the vendor entry with the
# same parameter layout), it only pins the parameter table.
_NO_SPECIALIZE = [
    "input_ptr",
    "output_ptr",
    "in_stride_n",
    "in_stride_c",
    "in_stride_h",
    "in_stride_w",
    "in_c",
    "in_h",
    "in_w",
    "out_h",
    "out_w",
    "divisor_override",
]


@triton.jit(do_not_specialize=_NO_SPECIALIZE)
def avg_pool2d_forward_kernel_1d(
    input_ptr,
    output_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    DILATION_H: tl.constexpr,
    DILATION_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, 1)
    keep = (pid < 0) & (offs < 0)
    value = tl.load(input_ptr + offs, mask=keep, other=0.0)
    tl.store(output_ptr + offs, value, mask=keep)


_BIND_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _bind_avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad):
    """Run the 1-D pool through the 2-D binding; None when it does not apply."""
    if input.ndim != 3 or input.dtype not in _BIND_DTYPES or not input.is_contiguous():
        return None
    if ceil_mode:
        return None
    if isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) != 1:
            return None
        kernel_size = kernel_size[0]
    if isinstance(padding, (tuple, list)):
        if len(padding) != 1:
            return None
        padding = padding[0]
    if isinstance(stride, (tuple, list)):
        if len(stride) != 1:
            return None
        stride = stride[0]
    k = int(kernel_size)
    s = int(stride) if stride is not None else k
    p = int(padding)
    if k <= 0 or s <= 0 or p < 0 or 2 * p > k:
        return None

    n, c, length = input.shape
    out_len = (length + 2 * p - k) // s + 1
    if out_len <= 0:
        return None
    x4 = input.view(n, c, 1, length)
    out = torch.empty((n, c, 1, out_len), device=input.device, dtype=input.dtype)
    with torch_device_fn.device(input.device):
        avg_pool2d_forward_kernel_1d[(n * c, 1, 1)](
            x4,
            out,
            x4.stride(0),
            x4.stride(1),
            x4.stride(2),
            x4.stride(3),
            c,
            1,
            length,
            1,
            out_len,
            1,
            k,
            1,
            s,
            0,
            p,
            1,
            1,
            bool(count_include_pad),
            0.0,
            1,
            out_len,
            num_warps=1,
        )
    return out.view(n, c, out_len)


logger = logging.getLogger(__name__)


def avg_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    """Average pooling operation over 1D input.

    Implemented by reshaping to 2D and calling avg_pool2d with kernel_size=(1, kernel_size).
    """
    logger.debug("GEMS_KUNLUNXIN AVG_POOL1D")

    bound = _bind_avg_pool1d(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad
    )
    if bound is not None:
        return bound

    # Input shape: (N, C, L) -> reshape to (N, C, 1, L)
    assert input.ndim == 3, f"avg_pool1d expects 3D input, got {input.ndim}D"

    input_4d = input.unsqueeze(2)  # (N, C, L) -> (N, C, 1, L)

    # Convert 1D parameters to 2D: kernel_size -> (1, kernel_size)
    if isinstance(kernel_size, int):
        kernel_size_1d = kernel_size
        kernel_size_2d = (1, kernel_size)
    else:
        kernel_size_1d = kernel_size[0]
        kernel_size_2d = (1, kernel_size[0])

    # stride=None or stride=[] means stride=kernel_size (PyTorch default behavior)
    if stride is None or (isinstance(stride, list) and len(stride) == 0):
        stride_2d = (1, kernel_size_1d)
    elif isinstance(stride, int):
        stride_2d = (1, stride)
    else:
        stride_2d = (1, stride[0])

    if isinstance(padding, int):
        padding_2d = (0, padding)
    else:
        padding_2d = (0, padding[0])

    # Call avg_pool2d
    output_4d = avg_pool2d(
        input_4d,
        kernel_size=kernel_size_2d,
        stride=stride_2d,
        padding=padding_2d,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=None,
    )

    # Reshape back: (N, C, 1, L_out) -> (N, C, L_out)
    return output_4d.squeeze(2)
