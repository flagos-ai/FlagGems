# Copyright 2024 FlagGems Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['num_elements'],
)
@triton.jit
def upsample_nearest2d_kernel(
    inp_ptr, out_ptr,
    num_elements,
    B, C, H_in, W_in, H_out, W_out,
    scale_h, scale_w,
    s_b_in, s_c_in, s_h_in, s_w_in,
    s_b_out, s_c_out, s_h_out, s_w_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_elements
    is_nhwc = (s_c_out == 1)
    
    if is_nhwc:
        tmp_idx = idx
        curr_c = tmp_idx % C
        tmp_idx //= C
        curr_w_out = tmp_idx % W_out
        tmp_idx //= W_out
        curr_h_out = tmp_idx % H_out
        curr_b = tmp_idx // H_out
    else:
        tmp_idx = idx
        curr_w_out = tmp_idx % W_out
        tmp_idx //= W_out
        curr_h_out = tmp_idx % H_out
        tmp_idx //= H_out
        curr_c = tmp_idx % C
        curr_b = tmp_idx // C

    curr_h_in = (curr_h_out.to(tl.float32) / scale_h).to(tl.int32)
    curr_w_in = (curr_w_out.to(tl.float32) / scale_w).to(tl.int32)
    curr_h_in = tl.where(curr_h_in < H_in, curr_h_in, H_in - 1)
    curr_w_in = tl.where(curr_w_in < W_in, curr_w_in, W_in - 1)

    inp_offset = (curr_b * s_b_in + curr_c * s_c_in + 
                  curr_h_in * s_h_in + curr_w_in * s_w_in)
    tl.store(out_ptr + idx, tl.load(inp_ptr + inp_offset, mask=mask), mask=mask)

@triton.jit
def upsample_nearest2d_backward_kernel(
    grad_out_ptr, grad_in_ptr,
    num_elements, 
    B, C, H_in, W_in, H_out, W_out,
    scale_h, scale_w,
    s_b_out, s_c_out, s_h_out, s_w_out,
    s_b_in, s_c_in, s_h_in, s_w_in,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_elements

    tmp_idx = idx
    curr_w_out = tmp_idx % W_out
    tmp_idx //= W_out
    curr_h_out = tmp_idx % H_out
    tmp_idx //= H_out
    curr_c = tmp_idx % C
    curr_b = tmp_idx // C

    curr_h_in = (curr_h_out.to(tl.float32) / scale_h).to(tl.int32)
    curr_w_in = (curr_w_out.to(tl.float32) / scale_w).to(tl.int32)
    curr_h_in = tl.where(curr_h_in < H_in, curr_h_in, H_in - 1)
    curr_w_in = tl.where(curr_w_in < W_in, curr_w_in, W_in - 1)

    out_offset = (curr_b * s_b_out + curr_c * s_c_out + 
                  curr_h_out * s_h_out + curr_w_out * s_w_out)
    grad_in_offset = (curr_b * s_b_in + curr_c * s_c_in + 
                      curr_h_in * s_h_in + curr_w_in * s_w_in)
    
    val = tl.load(grad_out_ptr + out_offset, mask=mask)
    tl.atomic_add(grad_in_ptr + grad_in_offset, val, mask=mask)

class TritonUpsampleNearest2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output_size):
        B, C, H_in, W_in = input.shape
        H_out, W_out = output_size
        scale_h, scale_w = H_out / H_in, W_out / W_in
        ctx.output_size, ctx.input_shape, ctx.scales = output_size, input.shape, (scale_h, scale_w)
        mem_fmt = torch.channels_last if input.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
        out = torch.empty((B, C, H_out, W_out), device=input.device, dtype=input.dtype, memory_format=mem_fmt)
        num_el = out.numel()
        grid = lambda META: (triton.cdiv(num_el, META['BLOCK_SIZE']),)
        upsample_nearest2d_kernel[grid](input, out, num_el, B, C, H_in, W_in, H_out, W_out, scale_h, scale_w, *input.stride(), *out.stride())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        B, C, H_in, W_in = ctx.input_shape
        H_out, W_out = ctx.output_size
        scale_h, scale_w = ctx.scales
        grad_input = torch.zeros(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
        num_el = grad_output.numel()
        grid = (triton.cdiv(num_el, 1024),)
        upsample_nearest2d_backward_kernel[grid](grad_output, grad_input, num_el, B, C, H_in, W_in, H_out, W_out, scale_h, scale_w, *grad_output.stride(), *grad_input.stride(), BLOCK_SIZE=1024)
        return grad_input, None

def upsample_nearest2d(input, size):
    if isinstance(size, int): size = (size, size)
    return TritonUpsampleNearest2d.apply(input, size)