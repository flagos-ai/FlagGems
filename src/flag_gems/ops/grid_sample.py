import logging
import torch
import triton
import triton.language as tl
from flag_gems import runtime
from flag_gems.utils import libentry

@triton.jit
def get_cubic_weight(x, a):
    # Keys' cubic interpolation kernel with a = -0.75
    abs_x = tl.abs(x)
    mask1 = abs_x <= 1.0
    mask2 = (abs_x > 1.0) & (abs_x < 2.0)
    w1 = ((a + 2.0) * abs_x - (a + 3.0)) * abs_x * abs_x + 1.0
    w2 = ((a * abs_x - 5.0 * a) * abs_x + 8.0 * a) * abs_x - 4.0 * a
    return tl.where(mask1, w1, tl.where(mask2, w2, 0.0))

@libentry()
@triton.jit
def grid_sample_2d_bicubic_kernel(
    output_ptr, input_ptr, grid_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    padding_mode: tl.constexpr, # 0: zeros, 1: border, 2: reflection
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * H_out * W_out)

    w_out_idx = offsets % W_out
    tmp = offsets // W_out
    h_out_idx = tmp % H_out
    n_idx = tmp // H_out

    grid_offset = n_idx * (H_out * W_out * 2) + h_out_idx * (W_out * 2) + w_out_idx * 2
    gx = tl.load(grid_ptr + grid_offset, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1, mask=mask, other=0.0)

    if align_corners:
        ix = ((gx + 1) / 2) * (W_in - 1)
        iy = ((gy + 1) / 2) * (H_in - 1)
    else:
        ix = ((gx + 1) * W_in - 1) / 2
        iy = ((gy + 1) * H_in - 1) / 2

    ix_floor = tl.floor(ix)
    iy_floor = tl.floor(iy)
    x_start = ix_floor - 1.0
    y_start = iy_floor - 1.0
    cubic_a = -0.75

    for c in range(C):
        channel_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        input_c_ptr = input_ptr + (n_idx * C * H_in * W_in) + (c * H_in * W_in)
        
        for dy in range(4):
            curr_y = y_start + dy
            wy = get_cubic_weight(iy - curr_y, cubic_a)
            y_idx = tl.cast(curr_y, tl.int32)
            y_valid = (y_idx >= 0) & (y_idx < H_in)
            
            if padding_mode == 1: # Border
                y_idx = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
                y_valid = True 
            elif padding_mode == 2: # Reflection
                 y_idx = tl.maximum(0, tl.minimum(y_idx, H_in - 1))
                 y_valid = True

            for dx in range(4):
                curr_x = x_start + dx
                wx = get_cubic_weight(ix - curr_x, cubic_a)
                x_idx = tl.cast(curr_x, tl.int32)
                x_valid = (x_idx >= 0) & (x_idx < W_in)
                
                if padding_mode == 1: # Border
                    x_idx = tl.maximum(0, tl.minimum(x_idx, W_in - 1))
                    x_valid = True
                
                pixel_offset = y_idx * W_in + x_idx
                val = tl.load(input_c_ptr + pixel_offset, mask=mask & y_valid & x_valid, other=0.0)
                channel_val += val * (wx * wy)

        out_offset = (n_idx * C * H_out * W_out) + (c * H_out * W_out) + (h_out_idx * W_out) + w_out_idx
        tl.store(output_ptr + out_offset, channel_val, mask=mask)

@libentry()
@triton.jit
def grid_sample_2d_nearest_kernel(
    output_ptr, input_ptr, grid_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * H_out * W_out)

    w_out_idx = offsets % W_out
    tmp = offsets // W_out
    h_out_idx = tmp % H_out
    n_idx = tmp // H_out

    grid_offset = n_idx * (H_out * W_out * 2) + h_out_idx * (W_out * 2) + w_out_idx * 2
    gx = tl.load(grid_ptr + grid_offset, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1, mask=mask, other=0.0)

    if align_corners:
        ix = ((gx + 1) / 2) * (W_in - 1)
        iy = ((gy + 1) / 2) * (H_in - 1)
        ix = tl.libdevice.round(ix)
        iy = tl.libdevice.round(iy)
    else:
        ix = ((gx + 1) * W_in - 1) / 2
        iy = ((gy + 1) * H_in - 1) / 2
        ix = tl.libdevice.round(ix)
        iy = tl.libdevice.round(iy)

    ix = tl.cast(ix, tl.int32)
    iy = tl.cast(iy, tl.int32)

    valid = (ix >= 0) & (ix < W_in) & (iy >= 0) & (iy < H_in)

    for c in range(C):
        input_val = tl.load(
            input_ptr + n_idx * C * H_in * W_in + c * H_in * W_in + iy * W_in + ix,
            mask=mask & valid,
            other=0.0
        )
        out_offset = n_idx * C * H_out * W_out + c * H_out * W_out + h_out_idx * W_out + w_out_idx
        tl.store(output_ptr + out_offset, input_val, mask=mask)

@libentry()
@triton.jit
def grid_sample_2d_bilinear_kernel(
    output_ptr, input_ptr, grid_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * H_out * W_out)

    w_out_idx = offsets % W_out
    tmp = offsets // W_out
    h_out_idx = tmp % H_out
    n_idx = tmp // H_out

    grid_offset = n_idx * (H_out * W_out * 2) + h_out_idx * (W_out * 2) + w_out_idx * 2
    gx = tl.load(grid_ptr + grid_offset, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1, mask=mask, other=0.0)

    if align_corners:
        ix = ((gx + 1) / 2) * (W_in - 1)
        iy = ((gy + 1) / 2) * (H_in - 1)
    else:
        ix = ((gx + 1) * W_in - 1) / 2
        iy = ((gy + 1) * H_in - 1) / 2

    ix_nw = tl.floor(ix)
    iy_nw = tl.floor(iy)
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

    for c in range(C):
        base_ptr = input_ptr + n_idx * C * H_in * W_in + c * H_in * W_in
        
        # NW
        val_nw = tl.load(base_ptr + iy_nw * W_in + ix_nw, mask=mask & (ix_nw >= 0) & (ix_nw < W_in) & (iy_nw >= 0) & (iy_nw < H_in), other=0.0)
        # NE
        val_ne = tl.load(base_ptr + iy_ne * W_in + ix_ne, mask=mask & (ix_ne >= 0) & (ix_ne < W_in) & (iy_ne >= 0) & (iy_ne < H_in), other=0.0)
        # SW
        val_sw = tl.load(base_ptr + iy_sw * W_in + ix_sw, mask=mask & (ix_sw >= 0) & (ix_sw < W_in) & (iy_sw >= 0) & (iy_sw < H_in), other=0.0)
        # SE
        val_se = tl.load(base_ptr + iy_se * W_in + ix_se, mask=mask & (ix_se >= 0) & (ix_se < W_in) & (iy_se >= 0) & (iy_se < H_in), other=0.0)

        result = val_nw * nw + val_ne * ne + val_sw * sw + val_se * se
        out_offset = n_idx * C * H_out * W_out + c * H_out * W_out + h_out_idx * W_out + w_out_idx
        tl.store(output_ptr + out_offset, result, mask=mask)

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    N, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
    
    grid_meta = lambda META: (triton.cdiv(N * H_out * W_out, META['BLOCK_SIZE']),)
    
    pad_enum = 0
    if padding_mode == 'border': pad_enum = 1
    elif padding_mode == 'reflection': pad_enum = 2

    if mode == 'nearest':
        grid_sample_2d_nearest_kernel[grid_meta](
            output, input, grid, N, C, H_in, W_in, H_out, W_out, align_corners, BLOCK_SIZE=256
        )
    elif mode == 'bicubic':
        grid_sample_2d_bicubic_kernel[grid_meta](
            output, input, grid, N, C, H_in, W_in, H_out, W_out, align_corners, pad_enum, BLOCK_SIZE=256
        )
    else:
        grid_sample_2d_bilinear_kernel[grid_meta](
            output, input, grid, N, C, H_in, W_in, H_out, W_out, align_corners, BLOCK_SIZE=256
        )
    return output