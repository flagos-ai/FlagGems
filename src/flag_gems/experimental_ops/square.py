import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def square_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * x
    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_square(x: torch.Tensor, out: torch.Tensor):
    n_elements = out.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    square_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)


def square(input: torch.Tensor, *, out: Optional[torch.Tensor] = None):
    assert isinstance(input, torch.Tensor), "input must be a torch.Tensor"
    assert input.is_cuda, "input must be on a CUDA device"
    if out is None:
        out = torch.empty_like(input)
    else:
        assert isinstance(out, torch.Tensor), "out must be a torch.Tensor"
        assert out.is_cuda, "out must be on a CUDA device"
        assert out.dtype == input.dtype, "out dtype must match input dtype"
        assert out.numel() == input.numel(), "out must have the same number of elements as input"
        assert out.shape == input.shape, "out must have the same shape as input"

    x_c = input.contiguous()
    out_c = out.contiguous()

    _launch_square(x_c, out_c)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def square_out(input: torch.Tensor, out: torch.Tensor):
    assert isinstance(input, torch.Tensor), "input must be a torch.Tensor"
    assert isinstance(out, torch.Tensor), "out must be a torch.Tensor"
    assert input.is_cuda and out.is_cuda, "input and out must be on a CUDA device"
    assert out.dtype == input.dtype, "out dtype must match input dtype"
    assert out.numel() == input.numel(), "out must have the same number of elements as input"
    assert out.shape == input.shape, "out must have the same shape as input"

    x_c = input.contiguous()
    out_c = out.contiguous()

    _launch_square(x_c, out_c)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out