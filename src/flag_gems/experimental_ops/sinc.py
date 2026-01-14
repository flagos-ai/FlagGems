import torch
import triton
import triton.language as tl


@triton.jit
def sinc_kernel_fp32(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)  # fp32
    y = x * 3.141592653589793
    siny = tl.sin(y)
    val = siny / y
    out = tl.where(x == 0.0, 1.0, val)

    tl.store(out_ptr + offsets, out, mask=mask)


def sinc(input: torch.Tensor):
    x_fp32 = input.contiguous().to(torch.float32)
    out_fp32 = torch.empty_like(x_fp32)
    n_elements = x_fp32.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sinc_kernel_fp32[grid](x_fp32, out_fp32, n_elements, BLOCK_SIZE=1024)

    if input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return out_fp32.to(input.dtype)
    else:
        return out_fp32


def sinc_out(input: torch.Tensor, out: torch.Tensor):
    x_fp32 = input.contiguous().to(torch.float32)
    out_fp32 = torch.empty_like(x_fp32)
    n_elements = x_fp32.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sinc_kernel_fp32[grid](x_fp32, out_fp32, n_elements, BLOCK_SIZE=1024)

    out.copy_(out_fp32.to(out.dtype))
    return out
