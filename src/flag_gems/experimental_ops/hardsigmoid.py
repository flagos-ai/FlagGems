import torch
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    y = xf * (1.0 / 6.0) + 0.5
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = y.to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


def hardsigmoid(x: torch.Tensor):
    out = torch.empty_like(x)
    assert x.is_cuda and out.is_cuda
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def hardsigmoid_out(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda
    assert x.numel() == out.numel()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out
