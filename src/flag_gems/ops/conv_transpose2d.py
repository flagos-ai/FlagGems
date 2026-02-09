# Copyright 2024 FlagGems Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2),
    ],
    key=['N', 'C_in', 'C_out', 'H_out', 'W_out'],
)
@triton.jit
def conv_transpose2d_nhwc_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    s_n_in, s_h_in, s_w_in, 
    s_c_out_w, s_kh_w, s_kw_w, 
    s_n_out, s_h_out, s_w_out,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr, 
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N * H_out * W_out, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    ofs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = ofs_m < (N * H_out * W_out)

    tmp = ofs_m
    w_out = tmp % W_out
    tmp = tmp // W_out
    h_out = tmp % H_out
    n = tmp // H_out

    ofs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = ofs_n < C_out

    # Accumulate in FP32 regardless of input type
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + ofs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = tl.broadcast_to(bias_val[None, :], (BLOCK_M, BLOCK_N))
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_h in range(K_h):
        for k_w in range(K_w):
            h_val = h_out + pad_h - k_h * dilation_h
            w_val = w_out + pad_w - k_w * dilation_w
            
            valid_h = (h_val % stride_h == 0)
            valid_w = (w_val % stride_w == 0)
            
            h_in = h_val // stride_h
            w_in = w_val // stride_w
            
            bounds_mask = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            geo_mask = mask_m & bounds_mask & valid_h & valid_w
            
            base_in_off = n * s_n_in + h_in * s_h_in + w_in * s_w_in
            base_w_off = ofs_n[None, :] * s_c_out_w + k_h * s_kh_w + k_w * s_kw_w

            for c_in_start in range(0, C_in, BLOCK_K):
                ofs_k = c_in_start + tl.arange(0, BLOCK_K)
                mask_k = ofs_k < C_in
                
                in_off = base_in_off[:, None] + ofs_k[None, :] 
                w_off = base_w_off + ofs_k[:, None] 

                a = tl.load(input_ptr + in_off, mask=geo_mask[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(weight_ptr + w_off, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32).to(tl.float32)

    out_off = n[:, None] * s_n_out + h_out[:, None] * s_h_out + w_out[:, None] * s_w_out + ofs_n[None, :]
    tl.store(output_ptr + out_off, acc, mask=mask_m[:, None] & mask_n[None, :])

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, allow_tf32=True):
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)
    if isinstance(output_padding, int): output_padding = (output_padding, output_padding)
    
    N, C_in, H_in, W_in = input.shape
    C_in_w, C_out_group, K_h, K_w = weight.shape
    C_out = C_out_group * groups
    
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_h - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_w - 1) + output_padding[1] + 1
    
    if input.stride(-1) != 1:
        input = input.contiguous(memory_format=torch.channels_last)
        
    weight_nhwc = weight.permute(1, 2, 3, 0).contiguous()
    
    # -----------------------------------------------------------
    # FIXED: Use input.dtype so float16 input gets float16 output
    # -----------------------------------------------------------
    output = torch.empty((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype, memory_format=torch.channels_last).zero_()
    bias_ptr = bias if bias is not None else None
    
    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META['BLOCK_M']) * triton.cdiv(C_out, META['BLOCK_N']),
    )
    
    conv_transpose2d_nhwc_kernel[grid](
        input, weight_nhwc, bias_ptr, output,
        N, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        input.stride(0), input.stride(2), input.stride(3),
        weight_nhwc.stride(0), weight_nhwc.stride(1), weight_nhwc.stride(2),
        output.stride(0), output.stride(2), output.stride(3),
        ALLOW_TF32=allow_tf32
    )
    return output