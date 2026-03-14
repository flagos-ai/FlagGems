import torch
import triton
import triton.language as tl
from ..utils import pointwise_dynamic

@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16),
    ],
    key=["n_elements"],
)
@triton.jit
def leaky_relu_kernel(x_ptr, y_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x >= 0, x, x * negative_slope)
    tl.store(y_ptr + offsets, y, mask=mask)

def leaky_relu(input, negative_slope=0.01):
    out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    leaky_relu_kernel[grid](input, out, n_elements, negative_slope)
    return out

def leaky_relu_out(input, negative_slope=0.01, out=None):
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    leaky_relu_kernel[grid](input, out, n_elements, negative_slope)
    return out
