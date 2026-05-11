import torch
import triton
import triton.language as tl

@triton.jit
def max_pool3d_kernel(x_ptr, out_ptr, D, H, W, oD, oH, oW, k, s, p, stride_c, stride_d, stride_h, stride_w):
    idx = tl.program_id(0)
    pd = tl.program_id(1)
    hw_idx = tl.program_id(2)
    ph = hw_idx // oW
    pw = hw_idx % oW
    
    x_base = x_ptr + idx * stride_c
    max_val = -float('inf')
    for i in range(3): 
        for j in range(3):
            for m in range(3):
                val = tl.load(x_base + (pd*s-p+i)*stride_d + (ph*s-p+j)*stride_h + (pw*s-p+m)*stride_w, 
                               mask=((pd*s-p+i)>=0)&((pd*s-p+i)<D)&((ph*s-p+j)>=0)&((ph*s-p+j)<H)&((pw*s-p+m)>=0)&((pw*s-p+m)<W), 
                               other=-float('inf'))
                max_val = tl.maximum(max_val, val)
    tl.store(out_ptr + idx*(oD*oH*oW) + pd*(oH*oW) + ph*oW + pw, max_val)

def max_pool3d(x):
    N, C, D, H, W = x.shape
    oD, oH, oW = (D+2-3)//2+1, (H+2-3)//2+1, (W+2-3)//2+1
    out = torch.empty((N, C, oD, oH, oW), device=x.device)
    grid = (N*C, oD, oH * oW)
    max_pool3d_kernel[grid](x, out, D, H, W, oD, oH, oW, 3, 2, 1, x.stride(1), x.stride(2), x.stride(3), x.stride(4))
    return out
