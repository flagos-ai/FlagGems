import torch
import triton
import triton.language as tl


@triton.jit
def negative_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, -x, mask=mask)


def _launch_negative(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"

    n_elements = x.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    negative_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def negative(x: torch.Tensor):
    out = torch.empty_like(x.contiguous())
    _launch_negative(x.contiguous(), out)
    return out


def negative_out(x: torch.Tensor, out: torch.Tensor):
    _launch_negative(x, out)
    return out
