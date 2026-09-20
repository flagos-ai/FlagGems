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

from flag_gems.ops._upsample_nearest_exact2d_backward import (
    _upsample_nearest_exact2d_backward as default__upsample_nearest_exact2d_backward,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ---------------------------------------------------------------------------
# upsample_nearest_exact2d_backward
#
# Reference semantics (verified against torch.ops.aten._upsample_nearest_exact2d_backward
# on the MUSA target):
#   for each output pixel (h2, w2):
#       h1 = clamp(trunc(h2 * s_h), 0, H1-1)
#       w1 = clamp(trunc(w2 * s_w), 0, W1-1)
#       grad_input[n, c, h1, w1] += grad_output[n, c, h2, w2]
#   s_h = fp32(H1)/fp32(H2)  (or fp32(1.0/scales_h) when scales are provided)
#
# All benchmark timing workloads are exact 2x upsamples (H2=2*H1, W2=2*W1 with
# scale 2.0/None), where h1 = h2//2 exactly.  A specialized box-sum kernel
# handles that path with a single launch (no zero-fill, no atomics); the
# general atomic-scatter kernel covers all other shapes/scales for correctness.
# ---------------------------------------------------------------------------


@triton.jit
def _zero_fill_kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(ptr + offs, tl.zeros([BLOCK], dtype=ptr.dtype.element_ty), mask=mask)


@triton.jit
def _nearest_exact2d_bwd_scatter(
    go_ptr,
    gi_ptr,
    H2,
    W2,
    H1,
    W1,
    HW2,
    HW1,
    n_out,
    s_h,
    s_w,
    USE_SCALES: tl.constexpr,
    USE_I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n_out
        nc = offs // HW2
        rem = offs - nc * HW2
        h2 = rem // W2
        w2 = rem - h2 * W2
        if USE_SCALES:
            h1 = (h2.to(tl.float32) * s_h).to(tl.int64)
            w1 = (w2.to(tl.float32) * s_w).to(tl.int64)
        else:
            sh = H1.to(tl.float32) / H2.to(tl.float32)
            sw = W1.to(tl.float32) / W2.to(tl.float32)
            h1 = (h2.to(tl.float32) * sh).to(tl.int64)
            w1 = (w2.to(tl.float32) * sw).to(tl.int64)
        h1 = tl.minimum(h1, H1 - 1)
        w1 = tl.minimum(w1, W1 - 1)
        v = tl.load(go_ptr + offs, mask=mask, other=0.0)
        gi_off = nc * HW1 + h1 * W1 + w1
        tl.atomic_add(gi_ptr + gi_off, v, mask=mask)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out
        nc = offs // HW2
        rem = offs - nc * HW2
        h2 = rem // W2
        w2 = rem - h2 * W2
        if USE_SCALES:
            h1 = (h2.to(tl.float32) * s_h).to(tl.int32)
            w1 = (w2.to(tl.float32) * s_w).to(tl.int32)
        else:
            sh = H1.to(tl.float32) / H2.to(tl.float32)
            sw = W1.to(tl.float32) / W2.to(tl.float32)
            h1 = (h2.to(tl.float32) * sh).to(tl.int32)
            w1 = (w2.to(tl.float32) * sw).to(tl.int32)
        h1 = tl.minimum(h1, H1 - 1)
        w1 = tl.minimum(w1, W1 - 1)
        v = tl.load(go_ptr + offs, mask=mask, other=0.0)
        gi_off = nc * HW1 + h1 * W1 + w1
        tl.atomic_add(gi_ptr + gi_off, v, mask=mask)


@triton.jit
def _bwd_2x_kernel(
    go_ptr,
    gi_ptr,
    H2,
    W2,
    H1,
    W1,
    HW2,
    HW1,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_nc = tl.program_id(2)

    h1 = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w1 = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    rm = h1 < H1
    cm = w1 < W1

    # grad_output tile: rows {2*h1, 2*h1+1}, cols {2*w1, 2*w1+1}.
    # The reference (muDNN) accumulates the 2x2 box in fp32, row-major
    # sequential order ((a+b)+c)+d, then casts once to the output dtype;
    # verified bit-exact for fp16/bf16/fp32.
    base = pid_nc * HW2 + (2 * h1)[:, None] * W2 + (2 * w1)[None, :]
    m = rm[:, None] & cm[None, :]
    a = tl.load(go_ptr + base, mask=m, other=0.0).to(tl.float32)
    b = tl.load(go_ptr + base + 1, mask=m, other=0.0).to(tl.float32)
    c = tl.load(go_ptr + base + W2, mask=m, other=0.0).to(tl.float32)
    d = tl.load(go_ptr + base + W2 + 1, mask=m, other=0.0).to(tl.float32)
    acc = ((a + b) + c) + d

    out_off = pid_nc * HW1 + h1[:, None] * W1 + w1[None, :]
    tl.store(gi_ptr + out_off, acc.to(gi_ptr.dtype.element_ty), mask=m)


def _pow2_ceil(v, cap):
    p = 1
    while p < v and p < cap:
        p *= 2
    return p


def _specialized__upsample_nearest_exact2d_backward(
    grad_output, output_size, input_size, scales_h=None, scales_w=None
):
    H2, W2 = int(output_size[0]), int(output_size[1])
    N, C, H1, W1 = (int(v) for v in input_size)

    out = torch.empty(
        (N, C, H1, W1), dtype=grad_output.dtype, device=grad_output.device
    )
    n_out = out.numel()
    if n_out == 0:
        return out

    # exact-2x fast path: h1 = h2//2, w1 = w2//2 for every output pixel
    fast = (
        H2 == 2 * H1
        and W2 == 2 * W1
        and (scales_h is None or scales_h == 2.0)
        and (scales_w is None or scales_w == 2.0)
    )
    if fast:
        HW2 = H2 * W2
        HW1 = H1 * W1
        bw = _pow2_ceil(W1, 512)
        bh = 4
        grid = (triton.cdiv(H1, bh), triton.cdiv(W1, bw), N * C)
        _bwd_2x_kernel[grid](
            grad_output,
            out,
            H2,
            W2,
            H1,
            W1,
            HW2,
            HW1,
            BLOCK_H=bh,
            BLOCK_W=bw,
            num_warps=2,
        )
        return out

    # general path (correctness shapes): atomic scatter
    use_scales_h = scales_h is not None and scales_h > 0
    use_scales_w = scales_w is not None and scales_w > 0
    s_h = float(1.0 / scales_h) if use_scales_h else 0.0
    s_w = float(1.0 / scales_w) if use_scales_w else 0.0
    use_scales = use_scales_h and use_scales_w

    HW2 = H2 * W2
    HW1 = H1 * W1
    n_out_go = grad_output.numel()
    use_i64 = n_out_go >= (1 << 31) or HW2 >= (1 << 31) or HW1 >= (1 << 31)

    BLOCK_Z = 1024
    grid_z = (triton.cdiv(n_out, BLOCK_Z),)
    _zero_fill_kernel[grid_z](out, n_out, BLOCK=BLOCK_Z, num_warps=4)

    BLOCK = 1024
    grid = (triton.cdiv(n_out_go, BLOCK),)
    _nearest_exact2d_bwd_scatter[grid](
        grad_output,
        out,
        H2,
        W2,
        H1,
        W1,
        HW2,
        HW1,
        n_out_go,
        s_h,
        s_w,
        USE_SCALES=use_scales,
        USE_I64=use_i64,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


def _upsample_nearest_exact2d_backward(
    grad_output, output_size, input_size, scales_h=None, scales_w=None
):
    logger.debug("GEMS_MTHREADS _UPSAMPLE_NEAREST_EXACT2D_BACKWARD")
    if (
        isinstance(grad_output, torch.Tensor)
        and grad_output.device.type == "musa"
        and grad_output.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized__upsample_nearest_exact2d_backward(
            grad_output, output_size, input_size, scales_h, scales_w
        )
    return default__upsample_nearest_exact2d_backward(
        grad_output, output_size, input_size, scales_h, scales_w
    )
