import torch
import triton
import triton.language as tl
from flag_gems.utils.libentry import libentry


@libentry()
@triton.jit
def grid_sample_bilinear_kernel(
    input_ptr, grid_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    grid_stride_n, grid_stride_h, grid_stride_w, grid_stride_c,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    mode: tl.constexpr,  # 0: bilinear, 1: nearest
    padding_mode: tl.constexpr,  # 0: zeros, 1: border, 2: reflection
    align_corners: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Grid Sample Kernel 实现双线性/最近邻插值
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    out_offset_n = pid_n * output_stride_n
    out_offset_h = pid_h * output_stride_h
    out_offset_w = pid_w * output_stride_w
    
    grid_offset = pid_n * grid_stride_n + pid_h * grid_stride_h + pid_w * grid_stride_w
    x = tl.load(grid_ptr + grid_offset + 0 * grid_stride_c)
    y = tl.load(grid_ptr + grid_offset + 1 * grid_stride_c)
    
    if align_corners:
        x = ((x + 1.0) * (W_in - 1)) * 0.5
        y = ((y + 1.0) * (H_in - 1)) * 0.5
    else:
        x = ((x + 1.0) * W_in - 1.0) * 0.5
        y = ((y + 1.0) * H_in - 1.0) * 0.5
    
    if mode == 1:  # nearest
        x_int = tl.extra.cuda.libdevice.round(x).to(tl.int32)
        y_int = tl.extra.cuda.libdevice.round(y).to(tl.int32)
        
        if padding_mode == 0:
            mask = (x_int >= 0) & (x_int < W_in) & (y_int >= 0) & (y_int < H_in)
        elif padding_mode == 1:
            x_int = tl.maximum(0, tl.minimum(x_int, W_in - 1))
            y_int = tl.maximum(0, tl.minimum(y_int, H_in - 1))
            mask = True
        else:
            x_int = _reflect_padding(x_int, W_in)
            y_int = _reflect_padding(y_int, H_in)
            mask = True
        
        for c_offset in range(0, C, BLOCK_C):
            c_idx = c_offset + tl.arange(0, BLOCK_C)
            c_mask = c_idx < C
            in_offset = pid_n * input_stride_n + y_int * input_stride_h + x_int * input_stride_w
            
            if padding_mode == 0:
                val = tl.load(input_ptr + in_offset + c_idx * input_stride_c, mask=c_mask & mask, other=0.0)
            else:
                val = tl.load(input_ptr + in_offset + c_idx * input_stride_c, mask=c_mask)
            
            tl.store(output_ptr + out_offset_n + out_offset_h + out_offset_w + c_idx * output_stride_c, val, mask=c_mask)
    
    else:  # bilinear
        x0 = tl.floor(x).to(tl.int32)
        x1 = x0 + 1
        y0 = tl.floor(y).to(tl.int32)
        y1 = y0 + 1
        
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)
        
        if padding_mode == 0:
            mask_x0 = (x0 >= 0) & (x0 < W_in)
            mask_x1 = (x1 >= 0) & (x1 < W_in)
            mask_y0 = (y0 >= 0) & (y0 < H_in)
            mask_y1 = (y1 >= 0) & (y1 < H_in)
            mask_00 = mask_x0 & mask_y0
            mask_10 = mask_x1 & mask_y0
            mask_01 = mask_x0 & mask_y1
            mask_11 = mask_x1 & mask_y1
        elif padding_mode == 1:
            x0 = tl.maximum(0, tl.minimum(x0, W_in - 1))
            x1 = tl.maximum(0, tl.minimum(x1, W_in - 1))
            y0 = tl.maximum(0, tl.minimum(y0, H_in - 1))
            y1 = tl.maximum(0, tl.minimum(y1, H_in - 1))
            mask_00 = mask_10 = mask_01 = mask_11 = True
        else:
            x0 = _reflect_padding(x0, W_in)
            x1 = _reflect_padding(x1, W_in)
            y0 = _reflect_padding(y0, H_in)
            y1 = _reflect_padding(y1, H_in)
            mask_00 = mask_10 = mask_01 = mask_11 = True
        
        base_offset_n = pid_n * input_stride_n
        
        for c_offset in range(0, C, BLOCK_C):
            c_idx = c_offset + tl.arange(0, BLOCK_C)
            c_mask = c_idx < C
            c_stride = c_idx * input_stride_c
            
            offset_00 = base_offset_n + y0 * input_stride_h + x0 * input_stride_w + c_stride
            offset_10 = base_offset_n + y0 * input_stride_h + x1 * input_stride_w + c_stride
            offset_01 = base_offset_n + y1 * input_stride_h + x0 * input_stride_w + c_stride
            offset_11 = base_offset_n + y1 * input_stride_h + x1 * input_stride_w + c_stride
            
            if padding_mode == 0:
                v00 = tl.load(input_ptr + offset_00, mask=c_mask & mask_00, other=0.0)
                v10 = tl.load(input_ptr + offset_10, mask=c_mask & mask_10, other=0.0)
                v01 = tl.load(input_ptr + offset_01, mask=c_mask & mask_01, other=0.0)
                v11 = tl.load(input_ptr + offset_11, mask=c_mask & mask_11, other=0.0)
            else:
                v00 = tl.load(input_ptr + offset_00, mask=c_mask)
                v10 = tl.load(input_ptr + offset_10, mask=c_mask)
                v01 = tl.load(input_ptr + offset_01, mask=c_mask)
                v11 = tl.load(input_ptr + offset_11, mask=c_mask)
            
            out_val = wa * v00 + wb * v10 + wc * v01 + wd * v11
            tl.store(output_ptr + out_offset_n + out_offset_h + out_offset_w + c_stride, out_val, mask=c_mask)


@triton.jit
def _reflect_padding(coord, size):
    coord = tl.where(coord < 0, -coord, coord)
    period = 2 * (size - 1)
    if period > 0:
        coord = coord % period
        coord = tl.where(coord >= size, period - coord, coord)
    return coord


def grid_sample(input: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear", 
                padding_mode: str = "zeros", align_corners: bool = False) -> torch.Tensor:
    assert input.dim() == 4, f"grid_sample: 仅支持4D输入"
    assert grid.dim() == 4 and grid.size(-1) == 2, f"grid_sample: grid必须是(N,H,W,2)"
    
    N, C, H_in, W_in = input.shape
    N_g, H_out, W_out, _ = grid.shape
    assert N == N_g, f"batch size不匹配"
    
    output = torch.empty(N, C, H_out, W_out, dtype=input.dtype, device=input.device)
    
    mode_map = {"bilinear": 0, "nearest": 1}
    padding_map = {"zeros": 0, "border": 1, "reflection": 2}
    mode_val = mode_map[mode]
    padding_val = padding_map[padding_mode]
    
    if not input.is_contiguous():
        input = input.contiguous()
    if not grid.is_contiguous():
        grid = grid.contiguous()
    
    input_stride = input.stride()
    grid_stride = grid.stride()
    output_stride = output.stride()
    
    grid_blocks = (N, H_out, W_out)
    BLOCK_C = min(128, triton.next_power_of_2(C))
    
    grid_sample_bilinear_kernel[grid_blocks](
        input, grid, output,
        N, C, H_in, W_in, H_out, W_out,
        input_stride[0], input_stride[1], input_stride[2], input_stride[3],
        grid_stride[0], grid_stride[1], grid_stride[2], grid_stride[3],
        output_stride[0], output_stride[1], output_stride[2], output_stride[3],
        mode=mode_val, padding_mode=padding_val, align_corners=align_corners, BLOCK_C=BLOCK_C,
    )
    
    return output
