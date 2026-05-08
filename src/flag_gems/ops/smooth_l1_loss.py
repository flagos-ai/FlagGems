import torch
import triton
import triton.language as tl

@triton.jit
def smooth_l1_kernel(x_ptr, y_ptr, out_ptr, beta, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    diff = x - y
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    tl.store(out_ptr + offsets, loss, mask=mask)

def smooth_l1_triton(x, y, beta=1.0):
    n = x.numel()
    out = torch.empty_like(x)
    grid = ( (n + 1024 - 1) // 1024, )
    smooth_l1_kernel[grid](x, y, out, beta, n, 1024)
    return out
