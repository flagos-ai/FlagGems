import torch
import triton
import triton.language as tl


@triton.jit
def binary_add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(out_ptr + offs, c, mask=mask)


def binary_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 1024
    grid = triton.cdiv(n, BLOCK_SIZE)
    binary_add_kernel[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
