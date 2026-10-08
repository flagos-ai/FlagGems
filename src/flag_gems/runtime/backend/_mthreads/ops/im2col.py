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

from flag_gems.ops.im2col import _compute_output_dims, _parse_2tuple
from flag_gems.ops.im2col import im2col as default_im2col
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["rows_total", "L"],
)
@triton.jit
def im2col_kernel(
    x_ptr,  # *Pointer* to input tensor [N, C, H, W]
    out_ptr,  # *Pointer* to output tensor [N, C*kH*kW, outH*outW]
    N,
    C,
    H,
    W,
    kH,
    kW,
    dH,
    dW,
    pH,
    pW,
    sH,
    sW,
    outH,
    outW,
    rows_total,  # C * kH * kW
    L,  # outH * outW
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    num_row_tiles = tl.cdiv(rows_total, BLOCK_M)
    n = pid0 // num_row_tiles
    row_tile = pid0 % num_row_tiles

    row_offsets = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_rows = row_offsets < rows_total
    mask_cols = col_offsets < L

    k_area = kH * kW

    c_idx = row_offsets // k_area
    rem = row_offsets % k_area
    kh_idx = rem // kW
    kw_idx = rem % kW

    oh_vec = col_offsets // outW
    ow_vec = col_offsets % outW

    # Broadcast to [BLOCK_M, BLOCK_N]
    oh = oh_vec[None, :]
    ow = ow_vec[None, :]
    kh = kh_idx[:, None]
    kw = kw_idx[:, None]
    c = c_idx[:, None]

    ih = oh * sH - pH + kh * dH
    iw = ow * sW - pW + kw * dW

    in_h = (ih >= 0) & (ih < H)
    in_w = (iw >= 0) & (iw < W)
    in_bounds = in_h & in_w

    # Base offsets (int64 to avoid overflow on large tensors)
    base_in = (n.to(tl.int64) * C * H * W).to(tl.int64)
    base_out = (n.to(tl.int64) * rows_total * L).to(tl.int64)

    ptrs_in = (
        x_ptr + base_in + ((c.to(tl.int64) * H + ih.to(tl.int64)) * W + iw.to(tl.int64))
    )

    ptrs_out = (
        out_ptr
        + base_out
        + (row_offsets[:, None].to(tl.int64) * L + col_offsets[None, :].to(tl.int64))
    )

    mask = mask_rows[:, None] & mask_cols[None, :] & in_bounds

    vals = tl.load(ptrs_in, mask=mask, other=0)
    tl.store(ptrs_out, vals, mask=(mask_rows[:, None] & mask_cols[None, :]))


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    return True


def _launch_im2col(x, out, kH, kW, dH, dW, pH, pW, sH, sW):
    x = x.contiguous()
    out = out.contiguous()

    N, C, H, W = x.shape
    outH, outW = _compute_output_dims(H, W, kH, kW, dH, dW, pH, pW, sH, sW)
    rows_total = C * kH * kW
    L = outH * outW

    if rows_total == 0 or L == 0 or N == 0:
        return out

    grid = lambda META: (
        N * triton.cdiv(rows_total, META["BLOCK_M"]),
        triton.cdiv(L, META["BLOCK_N"]),
    )

    with torch_device_fn.device(out.device):
        im2col_kernel[grid](
            x,
            out,
            N,
            C,
            H,
            W,
            kH,
            kW,
            dH,
            dW,
            pH,
            pW,
            sH,
            sW,
            outH,
            outW,
            rows_total,
            L,
        )
    return out


def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    logger.debug("GEMS_MTHREADS IM2COL")
    if not _use_triton_kernel(input):
        return default_im2col(input, kernel_size, dilation, padding, stride)

    x = input
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        return default_im2col(input, kernel_size, dilation, padding, stride)

    kH, kW = _parse_2tuple(kernel_size, "kernel_size")
    dH, dW = _parse_2tuple(dilation, "dilation")
    pH, pW = _parse_2tuple(padding, "padding")
    sH, sW = _parse_2tuple(stride, "stride")

    N, C, H, W = x.shape
    outH, outW = _compute_output_dims(H, W, kH, kW, dH, dW, pH, pW, sH, sW)
    rows_total = C * kH * kW
    L = outH * outW

    out = torch.empty((N, rows_total, L), device=x.device, dtype=x.dtype)
    if L == 0 or rows_total == 0 or N == 0:
        return out if input.ndim == 4 else out.squeeze(0)

    _launch_im2col(x, out, kH, kW, dH, dW, pH, pW, sH, sW)
    return out if input.ndim == 4 else out.squeeze(0)
