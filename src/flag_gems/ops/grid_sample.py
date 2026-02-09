# Copyright 2024 FlagGems Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'H_out', 'W_out', 'C'],
)
@triton.jit
def grid_sample_2d_kernel(
    input_ptr, grid_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    s_in_n, s_in_c, s_in_h, s_in_w,
    s_grid_n, s_grid_h, s_grid_w, s_grid_co,
    s_out_n, s_out_c, s_out_h, s_out_w,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_spatial = N * H_out * W_out
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_spatial

    tmp_idx = idx
    w_out = tmp_idx % W_out
    tmp_idx //= W_out
    h_out = tmp_idx % H_out
    n = tmp_idx // H_out

    grid_off_x = n * s_grid_n + h_out * s_grid_h + w_out * s_grid_w + 0 * s_grid_co
    grid_off_y = n * s_grid_n + h_out * s_grid_h + w_out * s_grid_w + 1 * s_grid_co
    
    x = tl.load(grid_ptr + grid_off_x, mask=mask, other=0.0)
    y = tl.load(grid_ptr + grid_off_y, mask=mask, other=0.0)

    if align_corners:
        ix = ((x + 1) / 2) * (W_in - 1)
        iy = ((y + 1) / 2) * (H_in - 1)
    else:
        ix = ((x + 1) * W_in - 1) / 2
        iy = ((y + 1) * H_in - 1) / 2

    ix_nw = tl.floor(ix).to(tl.int32)
    iy_nw = tl.floor(iy).to(tl.int32)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_ne - ix) * (iy_sw - iy)
    ne = (ix - ix_nw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_nw)
    se = (ix - ix_nw) * (iy - iy_nw)

    check_nw = (ix_nw >= 0) & (ix_nw < W_in) & (iy_nw >= 0) & (iy_nw < H_in)
    check_ne = (ix_ne >= 0) & (ix_ne < W_in) & (iy_ne >= 0) & (iy_ne < H_in)
    check_sw = (ix_sw >= 0) & (ix_sw < W_in) & (iy_sw >= 0) & (iy_sw < H_in)
    check_se = (ix_se >= 0) & (ix_se < W_in) & (iy_se >= 0) & (iy_se < H_in)

    base_nw = n * s_in_n + iy_nw * s_in_h + ix_nw * s_in_w
    base_ne = n * s_in_n + iy_ne * s_in_h + ix_ne * s_in_w
    base_sw = n * s_in_n + iy_sw * s_in_h + ix_sw * s_in_w
    base_se = n * s_in_n + iy_se * s_in_h + ix_se * s_in_w
    
    base_out = n * s_out_n + h_out * s_out_h + w_out * s_out_w

    for c in range(C):
        off_c_in = c * s_in_c
        v_nw = tl.load(input_ptr + base_nw + off_c_in, mask=mask & check_nw, other=0.0)
        v_ne = tl.load(input_ptr + base_ne + off_c_in, mask=mask & check_ne, other=0.0)
        v_sw = tl.load(input_ptr + base_sw + off_c_in, mask=mask & check_sw, other=0.0)
        v_se = tl.load(input_ptr + base_se + off_c_in, mask=mask & check_se, other=0.0)
        val = v_nw * nw + v_ne * ne + v_sw * sw + v_se * se
        tl.store(output_ptr + base_out + c * s_out_c, val, mask=mask)

def grid_sample(input, grid, align_corners=False):
    N, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
    num_spatial = N * H_out * W_out
    grid_launch = lambda META: (triton.cdiv(num_spatial, META['BLOCK_SIZE']),)
    
    grid_sample_2d_kernel[grid_launch](
        input, grid, output,
        N, C, H_in, W_in, H_out, W_out,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        grid.stride(0), grid.stride(1), grid.stride(2), grid.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        align_corners=align_corners
    )
    return output