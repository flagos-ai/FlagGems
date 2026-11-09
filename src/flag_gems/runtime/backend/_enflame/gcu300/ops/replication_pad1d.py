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
#
# GCU300 specialization of replication_pad1d.
#
# The generic kernel computes lane-varying int64 pointer offsets.  On this
# target the int64 address arithmetic is miscompiled once a program covers
# more than one warp (lanes >= 32 all read the same wrong address), and the
# 2D launch exhausts the hardware grid.y limit (255) for B * C >= 256.
# This version keeps every offset in int32 and flattens the launch to 1D.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def replication_pad1d_kernel(
    in_ptr,
    out_ptr,
    C: tl.constexpr,
    W_in,
    W_out,
    pad_left,
    in_stride_n,
    in_stride_c,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_w,
    n_w_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_w = pid % n_w_blocks
    pid_nc = pid // n_w_blocks

    n = pid_nc // C
    c = pid_nc % C

    off_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off_w < W_out

    # Replication pad: clamp the source index into [0, W_in - 1].
    # tl.where instead of tl.maximum/tl.minimum: integer maximum feeding a
    # pointer offset aborts the compiler on this target.
    w_in = off_w - pad_left
    w_in = tl.where(w_in > 0, w_in, 0)
    w_in = tl.where(w_in < W_in - 1, w_in, W_in - 1)

    base_in = n * in_stride_n + c * in_stride_c
    base_out = n * out_stride_n + c * out_stride_c

    ptrs_in = in_ptr + base_in + w_in * in_stride_w
    ptrs_out = out_ptr + base_out + off_w * out_stride_w

    x = tl.load(ptrs_in, mask=mask, other=0)
    tl.store(ptrs_out, x, mask=mask)


def _launch_replication_pad1d_kernel(input: torch.Tensor, padding, out: torch.Tensor):
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])
    if left < 0 or right < 0:
        raise ValueError("Padding values must be non-negative for replication_pad1d")

    dim = input.dim()
    if dim not in (2, 3):
        raise ValueError("replication_pad1d expects 2D (C, W) or 3D (N, C, W) input")

    if dim == 3:
        N, C, W_in = input.shape
        in_s_n, in_s_c, in_s_w = input.stride()
        out_s_n, out_s_c, out_s_w = out.stride()
        expected_out_shape = (N, C, W_in + left + right)
    else:
        C, W_in = input.shape
        N = 1
        in_s_c, in_s_w = input.stride()
        in_s_n = 0
        if out.dim() == 2:
            out_s_n = 0
            out_s_c, out_s_w = out.stride()
        elif out.dim() == 3:
            out_s_n, out_s_c, out_s_w = out.stride()
        else:
            raise ValueError("Output tensor has invalid dimensions")
        expected_out_shape = (C, W_in + left + right)

    W_out = W_in + left + right

    if tuple(out.shape) != expected_out_shape:
        raise ValueError(
            f"Output tensor has incorrect shape. Expected {expected_out_shape}, got {tuple(out.shape)}"
        )

    BLOCK_SIZE = 256
    n_w_blocks = triton.cdiv(W_out, BLOCK_SIZE)
    grid = (n_w_blocks * N * C,)
    with torch_device_fn.device(input.device):
        replication_pad1d_kernel[grid](
            input,
            out,
            C,
            W_in,
            W_out,
            left,
            in_s_n,
            in_s_c,
            in_s_w,
            out_s_n,
            out_s_c,
            out_s_w,
            n_w_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out


def replication_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD1D")
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])
    if input.dim() == 3:
        N, C, W_in = input.shape
        out = torch.empty(
            (N, C, W_in + left + right),
            device=input.device,
            dtype=input.dtype,
            layout=input.layout,
        )
    elif input.dim() == 2:
        C, W_in = input.shape
        out = torch.empty(
            (C, W_in + left + right),
            device=input.device,
            dtype=input.dtype,
            layout=input.layout,
        )
    else:
        raise ValueError("replication_pad1d expects 2D (C, W) or 3D (N, C, W) input")
    return _launch_replication_pad1d_kernel(input, (left, right), out)


def replication_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD1D_OUT")
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])

    if out.dtype != input.dtype:
        raise ValueError("Output dtype must match input dtype")
    if out.device != input.device:
        raise ValueError("Output device must match input device")

    return _launch_replication_pad1d_kernel(input, (left, right), out)
