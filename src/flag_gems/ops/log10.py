import torch
import triton
import triton.language as tl

@triton.jit
def log10_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Formula: log10(x) = log2(x) * log10(2)
    # log10(2) is approximately 0.3010299956639812
    y = tl.log2(x) * 0.3010299956639812
    
    tl.store(y_ptr + offsets, y, mask=mask)

def log10(x: torch.Tensor):
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    log10_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y
