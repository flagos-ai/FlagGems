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
def _any_zero_partial_kernel(input_ptr, numel, partial_ptr, BLOCK: tl.constexpr):
    # Grid-parallel pass 1: each program reduces one chunk and flags whether any
    # element in it is zero.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    offs_c = tl.where(mask, offs, 0)
    v = tl.load(input_ptr + offs_c).to(tl.float32)
    v = tl.where(mask, v, 1.0)
    has_zero = tl.sum((v == 0.0).to(tl.int32), axis=0) > 0
    tl.store(partial_ptr + pid, has_zero.to(tl.int32))


@triton.jit
def _any_zero_final_kernel(partial_ptr, num_partial, out_ptr, BLOCK: tl.constexpr):
    # Pass 2: reduce the per-chunk flags to a single tensor-level flag.  This
    # mirrors aten::cumprod_backward's tensor-level branch: when any zero is
    # present it recomputes the cumprod, otherwise it reuses the forward output.
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(0, num_partial, BLOCK):
        idx = start + offs
        mask = idx < num_partial
        idx_c = tl.where(mask, idx, 0)
        v = tl.load(partial_ptr + idx_c).to(tl.int32)
        v = tl.where(mask, v, 0)
        acc += v
    has_zero = tl.sum(acc, axis=0) > 0
    tl.store(out_ptr, has_zero.to(tl.int32))


@triton.jit
def _cumprod_backward_kernel(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    has_zero_ptr,
    N,  # length along reduction dim
    stride_d,  # stride along reduction dim (same for all tensors)
    stride_b,  # batch stride (same for all tensors)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * stride_b

    offs = tl.arange(0, BLOCK)
    mask = offs < N
    # Clamp masked lanes to index 0 (this backend ignores tl.load(other=) and
    # would otherwise read neighboring memory into the reductions below).
    offs_clamped = tl.where(mask, offs, 0)

    g = tl.load(grad_ptr + base + offs_clamped * stride_d, cache_modifier=".ca").to(
        tl.float32
    )
    g = tl.where(mask, g, 0.0)
    inp = tl.load(input_ptr + base + offs_clamped * stride_d, cache_modifier=".ca").to(
        tl.float32
    )
    inp = tl.where(mask, inp, 0.0)
    out_bf16 = tl.load(
        output_ptr + base + offs_clamped * stride_d, cache_modifier=".ca"
    ).to(tl.float32)
    out_bf16 = tl.where(mask, out_bf16, 0.0)

    # Locate the first zero in this row (k = index, or large_val when there is none).
    is_zero = (inp == 0.0) & mask
    large_val = 2147483647
    zero_indices = tl.where(is_zero, offs.to(tl.int32), large_val)
    k = tl.min(zero_indices, axis=0)

    # Tensor-level zero flag: when any element is zero, aten recomputes the
    # cumprod in fp64 (so we recompute in fp32); otherwise it reuses the forward
    # output verbatim, which we must reproduce bit-for-bit to match the reference.
    has_zero = tl.load(has_zero_ptr).to(tl.int32)
    out_fp32 = tl.cumprod(inp, axis=0)
    out = tl.where(has_zero != 0, out_fp32, out_bf16)

    go = g * out

    # Reverse cumulative sum: rev_cs[i] = sum_{j=i}^{N-1} go[j]
    total_go = tl.sum(go, axis=0)
    inc_cs = tl.cumsum(go, axis=0)
    rev_cs = total_go - inc_cs + go

    safe_inp = tl.where(inp == 0.0, 1.0, inp)
    grad_input = tl.where(mask, rev_cs / safe_inp, 0.0)
    grad_input = tl.where(inp == 0.0, 0.0, grad_input)

    # Zero-handling is computed unconditionally (the tl.cumprod scan below must
    # NOT sit inside a runtime `if`, because ConvertTritonXPUToLLVM rejects
    # tt.scan inside a runtime scf.if on this backend).  When there is no zero,
    # k == large_val, so `offs == k` is false everywhere and the extra
    # cumprod/zero_val results are never selected.
    inp_mod = tl.where(offs.to(tl.int32) == k, 1.0, inp)
    inp_mod = tl.where(mask, inp_mod, 1.0)
    cumprod_mod = tl.cumprod(inp_mod, axis=0)

    gcm = g * cumprod_mod
    zero_val = tl.sum(tl.where(offs.to(tl.int32) >= k, gcm, 0.0), axis=0)

    grad_input = tl.where(offs.to(tl.int32) > k, 0.0, grad_input)
    grad_input = tl.where(offs.to(tl.int32) == k, zero_val, grad_input)

    tl.store(
        grad_input_ptr + base + offs * stride_d,
        grad_input.to(tl.float32),
        mask=mask,
        cache_modifier=".cs",
    )


@triton.jit
def _cumprod_backward_kernel_2d_batch(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    has_zero_ptr,
    N,
    stride_d,
    stride_b0,
    stride_b1,
    BLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = pid0 * stride_b0 + pid1 * stride_b1

    offs = tl.arange(0, BLOCK)
    mask = offs < N
    offs_clamped = tl.where(mask, offs, 0)

    g = tl.load(grad_ptr + base + offs_clamped * stride_d, cache_modifier=".ca").to(
        tl.float32
    )
    g = tl.where(mask, g, 0.0)
    inp = tl.load(input_ptr + base + offs_clamped * stride_d, cache_modifier=".ca").to(
        tl.float32
    )
    inp = tl.where(mask, inp, 0.0)
    out_bf16 = tl.load(
        output_ptr + base + offs_clamped * stride_d, cache_modifier=".ca"
    ).to(tl.float32)
    out_bf16 = tl.where(mask, out_bf16, 0.0)

    is_zero = (inp == 0.0) & mask
    large_val = 2147483647
    zero_indices = tl.where(is_zero, offs.to(tl.int32), large_val)
    k = tl.min(zero_indices, axis=0)

    has_zero = tl.load(has_zero_ptr).to(tl.int32)
    out_fp32 = tl.cumprod(inp, axis=0)
    out = tl.where(has_zero != 0, out_fp32, out_bf16)

    go = g * out

    total_go = tl.sum(go, axis=0)
    inc_cs = tl.cumsum(go, axis=0)
    rev_cs = total_go - inc_cs + go

    safe_inp = tl.where(inp == 0.0, 1.0, inp)
    grad_input = tl.where(mask, rev_cs / safe_inp, 0.0)
    grad_input = tl.where(inp == 0.0, 0.0, grad_input)

    inp_mod = tl.where(offs.to(tl.int32) == k, 1.0, inp)
    inp_mod = tl.where(mask, inp_mod, 1.0)
    cumprod_mod = tl.cumprod(inp_mod, axis=0)

    gcm = g * cumprod_mod
    zero_val = tl.sum(tl.where(offs.to(tl.int32) >= k, gcm, 0.0), axis=0)

    grad_input = tl.where(offs.to(tl.int32) > k, 0.0, grad_input)
    grad_input = tl.where(offs.to(tl.int32) == k, zero_val, grad_input)

    tl.store(
        grad_input_ptr + base + offs * stride_d,
        grad_input.to(tl.float32),
        mask=mask,
        cache_modifier=".cs",
    )


@triton.jit
def _cumprod_backward_kernel_3d_batch(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    has_zero_ptr,
    N,
    stride_d,
    stride_b0,
    stride_b1,
    stride_b2,
    BLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    base = pid0 * stride_b0 + pid1 * stride_b1 + pid2 * stride_b2

    offs = tl.arange(0, BLOCK)
    mask = offs < N
    offs_clamped = tl.where(mask, offs, 0)

    g = tl.load(grad_ptr + base + offs_clamped * stride_d, cache_modifier=".ca").to(
        tl.float32
    )
    g = tl.where(mask, g, 0.0)
    inp = tl.load(input_ptr + base + offs_clamped * stride_d, cache_modifier=".ca").to(
        tl.float32
    )
    inp = tl.where(mask, inp, 0.0)
    out_bf16 = tl.load(
        output_ptr + base + offs_clamped * stride_d, cache_modifier=".ca"
    ).to(tl.float32)
    out_bf16 = tl.where(mask, out_bf16, 0.0)

    is_zero = (inp == 0.0) & mask
    large_val = 2147483647
    zero_indices = tl.where(is_zero, offs.to(tl.int32), large_val)
    k = tl.min(zero_indices, axis=0)

    has_zero = tl.load(has_zero_ptr).to(tl.int32)
    out_fp32 = tl.cumprod(inp, axis=0)
    out = tl.where(has_zero != 0, out_fp32, out_bf16)

    go = g * out

    total_go = tl.sum(go, axis=0)
    inc_cs = tl.cumsum(go, axis=0)
    rev_cs = total_go - inc_cs + go

    safe_inp = tl.where(inp == 0.0, 1.0, inp)
    grad_input = tl.where(mask, rev_cs / safe_inp, 0.0)
    grad_input = tl.where(inp == 0.0, 0.0, grad_input)

    inp_mod = tl.where(offs.to(tl.int32) == k, 1.0, inp)
    inp_mod = tl.where(mask, inp_mod, 1.0)
    cumprod_mod = tl.cumprod(inp_mod, axis=0)

    gcm = g * cumprod_mod
    zero_val = tl.sum(tl.where(offs.to(tl.int32) >= k, gcm, 0.0), axis=0)

    grad_input = tl.where(offs.to(tl.int32) > k, 0.0, grad_input)
    grad_input = tl.where(offs.to(tl.int32) == k, zero_val, grad_input)

    tl.store(
        grad_input_ptr + base + offs * stride_d,
        grad_input.to(tl.float32),
        mask=mask,
        cache_modifier=".cs",
    )


def cumprod_backward(grad, input, dim, output):
    logger.debug("GEMS_KUNLUNXIN CUMPROD_BACKWARD")
    ndim = input.ndim
    dim = int(dim)
    if dim < 0:
        dim += ndim

    N = input.shape[dim]

    batch_size = 1
    for i in range(ndim):
        if i != dim:
            batch_size *= input.shape[i]

    grad_input = torch.empty_like(grad)

    grad_stride = grad.stride()
    input_stride = input.stride()
    output_stride = output.stride()
    gi_stride = grad_input.stride()

    batch_dims = [i for i in range(ndim) if i != dim]

    BLOCK = 16
    while BLOCK < N:
        BLOCK *= 2

    if BLOCK <= 32:
        num_warps = 1
    elif BLOCK <= 64:
        num_warps = 2
    elif BLOCK <= 128:
        num_warps = 4
    else:
        num_warps = 8

    # Tensor-level zero detection (mirrors aten::cumprod_backward, which branches
    # on whether any element is zero across the whole tensor).
    inp_flat = input.reshape(-1)
    if not inp_flat.is_contiguous():
        inp_flat = inp_flat.contiguous()
    numel = inp_flat.numel()
    has_zero_buf = torch.zeros(1, dtype=torch.int32, device=input.device)
    if numel > 0:
        RED_BLOCK = 4096
        num_blocks = (numel + RED_BLOCK - 1) // RED_BLOCK
        partial = torch.empty(num_blocks, dtype=torch.int32, device=input.device)
        with torch_device_fn.device(input.device):
            _any_zero_partial_kernel[(num_blocks,)](
                inp_flat, numel, partial, BLOCK=RED_BLOCK
            )
            _any_zero_final_kernel[(1,)](partial, num_blocks, has_zero_buf, BLOCK=1024)

    same_layout = (
        grad_stride == input_stride
        and grad_stride == output_stride
        and grad_stride == gi_stride
    )

    if not same_layout:
        grad_2d = grad.movedim(dim, -1).reshape(-1, N).contiguous()
        input_2d = input.movedim(dim, -1).reshape(-1, N).contiguous()
        output_2d = output.movedim(dim, -1).reshape(-1, N).contiguous()
        gi_2d = torch.empty((batch_size, N), dtype=grad.dtype, device=grad.device)

        grid = (batch_size,)
        with torch_device_fn.device(input.device):
            _cumprod_backward_kernel[grid](
                grad_2d,
                input_2d,
                output_2d,
                gi_2d,
                has_zero_buf,
                N,
                1,
                N,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        grad_input = (
            gi_2d.reshape([input.shape[i] for i in range(ndim) if i != dim] + [N])
            .movedim(-1, dim)
            .contiguous()
        )
        return grad_input

    stride_d = grad_stride[dim]

    with torch_device_fn.device(input.device):
        if len(batch_dims) == 0:
            grid = (1,)
            _cumprod_backward_kernel[grid](
                grad,
                input,
                output,
                grad_input,
                has_zero_buf,
                N,
                stride_d,
                0,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        elif len(batch_dims) == 1:
            stride_b = grad_stride[batch_dims[0]]
            grid = (batch_size,)
            _cumprod_backward_kernel[grid](
                grad,
                input,
                output,
                grad_input,
                has_zero_buf,
                N,
                stride_d,
                stride_b,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        elif len(batch_dims) == 2:
            stride_b0 = grad_stride[batch_dims[0]]
            stride_b1 = grad_stride[batch_dims[1]]
            s0 = input.shape[batch_dims[0]]
            s1 = input.shape[batch_dims[1]]
            grid = (s0, s1)
            _cumprod_backward_kernel_2d_batch[grid](
                grad,
                input,
                output,
                grad_input,
                has_zero_buf,
                N,
                stride_d,
                stride_b0,
                stride_b1,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        else:
            stride_b0 = grad_stride[batch_dims[0]]
            stride_b1 = grad_stride[batch_dims[1]]
            stride_b2 = grad_stride[batch_dims[2]]
            s0 = input.shape[batch_dims[0]]
            s1 = input.shape[batch_dims[1]]
            s2 = input.shape[batch_dims[2]]
            grid = (s0, s1, s2)
            _cumprod_backward_kernel_3d_batch[grid](
                grad,
                input,
                output,
                grad_input,
                has_zero_buf,
                N,
                stride_d,
                stride_b0,
                stride_b1,
                stride_b2,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )

    return grad_input
