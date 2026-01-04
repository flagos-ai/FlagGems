import torch
import triton
import triton.language as tl


@triton.jit
def hardshrink_kernel(x_ptr, out_ptr, n_elements, lambd, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    l = lambd
    keep = (x > l) | (x < -l)
    y = tl.where(keep, x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def _hardshrink_launch(x: torch.Tensor, lambd: float, out: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert out.is_cuda, "Output tensor must be on CUDA device"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"
    assert x.is_floating_point(), "hardshrink only supports floating point dtypes"

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    hardshrink_kernel[grid](x, out, n_elements, float(lambd), BLOCK_SIZE=BLOCK_SIZE)


def hardshrink(x: torch.Tensor, lambd: float = 0.5) -> torch.Tensor:
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _hardshrink_launch(x_c, lambd, out)
    return out


def hardshrink_out(
    x: torch.Tensor, lambd: float = 0.5, out: torch.Tensor = None
) -> torch.Tensor:
    x_c = x.contiguous()
    if out is None:
        out = torch.empty_like(x_c)
        _hardshrink_launch(x_c, lambd, out)
        return out
    # Ensure output is allocated correctly
    assert out.is_cuda, "Output tensor must be on CUDA device"
    assert out.dtype == x_c.dtype, "Output dtype must match input dtype"
    assert out.shape == x_c.shape, "Output shape must match input shape"

    if out.is_contiguous():
        _hardshrink_launch(x_c, lambd, out)
    else:
        tmp = torch.empty_like(x_c)
        _hardshrink_launch(x_c, lambd, tmp)
        out.copy_(tmp)
    return out
