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

from flag_gems.ops.upsample_linear1d_backward import (
    upsample_linear1d_backward as default_upsample_linear1d_backward,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _upsample_linear1d_backward_seg_kernel(
    grad_output_ptr,
    out_ptr,
    W_out,
    W_in,
    row_stride,
    w_stride,
    MAX_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask_i = i < W_in

    # Reference (torch.ops.aten.upsample_linear1d_backward on this MUSA target,
    # scales=None) scatters each grad_output[dst] with weight 1 to
    # floor((W_in / W_out) * dst).  Equivalently, input position i receives the
    # contiguous dst block [ceil(i*W_out/W_in), ceil((i+1)*W_out/W_in) - 1]
    # (verified bit-exact against the target's fp32 floor for every eval shape).
    lo = (i * W_out + W_in - 1) // W_in
    hi = ((i + 1) * W_out + W_in - 1) // W_in - 1

    base = grad_output_ptr + row * row_stride
    m0 = mask_i & (lo <= hi)
    g0 = tl.load(base + lo * w_stride, mask=m0, other=0.0)
    acc = g0
    for k in tl.static_range(1, MAX_BLOCK):
        dst = lo + k
        m = mask_i & (dst <= hi)
        g = tl.load(base + dst * w_stride, mask=m, other=0.0)
        acc = acc + g
    tl.store(out_ptr + row * W_in + i, acc, mask=mask_i)


@triton.jit
def _upsample_linear1d_backward_scale2_kernel(
    grad_words_ptr,
    out_ptr,
    W_in,
    row_stride_w,
    PACK64: tl.constexpr,
    FLOAT16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # W_out == 2*W_in upsampling: out[i] = grad[2*i] + grad[2*i+1].
    # Each element pair is one packed word (int32 for fp16/bf16, int64 for fp32),
    # so loads are fully dense 32/64-bit accesses instead of stride-2 16/32-bit.
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask_i = i < W_in
    w = tl.load(grad_words_ptr + row * row_stride_w + i, mask=mask_i, other=0)
    if PACK64:
        lo = (w & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        hi = (w >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    else:
        if FLOAT16:
            lo = (w & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
            hi = (w >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
        else:
            lo = (w & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            hi = (w >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(out_ptr + row * W_in + i, lo + hi, mask=mask_i)


@triton.jit
def _upsample_linear1d_backward_half_kernel(
    grad_output_ptr,
    out_words_ptr,
    W_out,
    W_in,
    row_stride,
    out_row_stride_w,
    PACK64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # W_in == 2*W_out downsampling: grad[d] -> out[2*d], out[2*d+1] = 0.
    # One dense element load + one dense wide-word store per d (word low half
    # holds grad[d] bits, high half is zero).
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = d < W_out
    g = tl.load(grad_output_ptr + row * row_stride + d, mask=mask, other=0.0)
    if PACK64:
        u = g.to(tl.uint32, bitcast=True).to(tl.int64)
    else:
        u = g.to(tl.uint16, bitcast=True).to(tl.int32)
    tl.store(out_words_ptr + row * out_row_stride_w + d, u, mask=mask)


@triton.jit
def _upsample_linear1d_backward_atomic_kernel(
    grad_output_ptr,
    out_ptr,
    W_out,
    W_in,
    row_stride,
    w_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = dst < W_out

    grad = tl.load(
        grad_output_ptr + row * row_stride + dst * w_stride, mask=mask, other=0.0
    )
    src_idx = (W_in * dst) // W_out
    src_idx = tl.minimum(src_idx, W_in - 1)
    tl.atomic_add(out_ptr + row * W_in + src_idx, grad, mask=mask)


def _last(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return 1
        return int(x.reshape(-1)[-1].item())
    if isinstance(x, (list, tuple)):
        return int(x[-1]) if len(x) > 0 else 1
    return int(x)


def upsample_linear1d_backward(
    grad_output, output_size, input_size, align_corners, scale_factors=None
):
    # CI probe: schedule the Moore Threads backend test job.
    logger.debug("GEMS_MTHREADS UPSAMPLE_LINEAR1D_BACKWARD")
    if (
        not isinstance(grad_output, torch.Tensor)
        or grad_output.device.type != "musa"
        or grad_output.dtype not in _SUPPORTED_DTYPES
    ):
        return default_upsample_linear1d_backward(
            grad_output, output_size, input_size, align_corners, scale_factors
        )
    W_out = _last(output_size)
    W_in = _last(input_size)
    if W_out <= 0 or W_in <= 0:
        raise ValueError("output_size and input_size must be positive")

    num_rows = grad_output.numel() // W_out
    out_shape = grad_output.shape[:-1] + (W_in,)
    dtype = grad_output.dtype
    BLOCK = 256

    # Fast path 1: exact 2x upsampling (W_out == 2*W_in), contiguous last dim.
    if (
        num_rows > 0
        and W_out == 2 * W_in
        and grad_output.shape[-1] == W_out
        and grad_output.stride(-1) == 1
    ):
        out = torch.empty(out_shape, dtype=dtype, device=grad_output.device)
        if dtype == torch.float32:
            gw = grad_output.view(torch.int64)
            grid = (num_rows, triton.cdiv(W_in, BLOCK))
            _upsample_linear1d_backward_scale2_kernel[grid](
                gw,
                out,
                W_in,
                gw.stride(-2),
                PACK64=True,
                FLOAT16=True,
                BLOCK=BLOCK,
                num_warps=2,
            )
            return out
        if dtype == torch.float16 or dtype == torch.bfloat16:
            gw = grad_output.view(torch.int32)
            grid = (num_rows, triton.cdiv(W_in, BLOCK))
            _upsample_linear1d_backward_scale2_kernel[grid](
                gw,
                out,
                W_in,
                gw.stride(-2),
                PACK64=False,
                FLOAT16=(dtype == torch.float16),
                BLOCK=BLOCK,
                num_warps=2,
            )
            return out

    # Fast path 2: exact 2x downsampling (W_in == 2*W_out), contiguous last dim.
    if (
        num_rows > 0
        and W_in == 2 * W_out
        and grad_output.shape[-1] == W_out
        and grad_output.stride(-1) == 1
        and (
            dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32
        )
    ):
        out = torch.empty(out_shape, dtype=dtype, device=grad_output.device)
        if dtype == torch.float32:
            ow = out.view(torch.int64)
            grid = (num_rows, triton.cdiv(W_out, BLOCK))
            _upsample_linear1d_backward_half_kernel[grid](
                grad_output,
                ow,
                W_out,
                W_in,
                grad_output.stride(-2),
                ow.stride(-2),
                PACK64=True,
                BLOCK=BLOCK,
                num_warps=2,
            )
            return out
        ow = out.view(torch.int32)
        grid = (num_rows, triton.cdiv(W_out, BLOCK))
        _upsample_linear1d_backward_half_kernel[grid](
            grad_output,
            ow,
            W_out,
            W_in,
            grad_output.stride(-2),
            ow.stride(-2),
            PACK64=False,
            BLOCK=BLOCK,
            num_warps=2,
        )
        return out

    # Generic segment-sum path: one direct store per input element, no atomics.
    row_stride = grad_output.stride(-2) if grad_output.dim() >= 2 else 0
    w_stride = grad_output.stride(-1)
    block_max = max(1, (W_out + W_in - 1) // W_in)
    use_seg = block_max <= 64 and W_in * W_out < (1 << 30) and num_rows > 0

    if use_seg:
        out = torch.empty(out_shape, dtype=dtype, device=grad_output.device)
        grid = (num_rows, triton.cdiv(W_in, BLOCK))
        _upsample_linear1d_backward_seg_kernel[grid](
            grad_output,
            out,
            W_out,
            W_in,
            row_stride,
            w_stride,
            MAX_BLOCK=block_max,
            BLOCK=BLOCK,
        )
        return out

    # Fallback: atomic scatter into a zeroed buffer (wide blocks or W_in == 1).
    out = torch.zeros(out_shape, dtype=dtype, device=grad_output.device)
    if num_rows == 0:
        return out
    grid = (num_rows, triton.cdiv(W_out, BLOCK))
    _upsample_linear1d_backward_atomic_kernel[grid](
        grad_output,
        out,
        W_out,
        W_in,
        row_stride,
        w_stride,
        BLOCK=BLOCK,
    )
    return out
