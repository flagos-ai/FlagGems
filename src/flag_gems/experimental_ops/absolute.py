import torch
import triton
import triton.language as tl


@triton.jit
def _absolute_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Generic absolute using branchless select: works for integers and floats.
    zero = x * 0
    is_neg = x < zero
    y = tl.where(is_neg, -x, x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _absolute_complex_kernel(ri_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ri_ptr points to the real-imag parts as a contiguous float tensor of shape (..., 2)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    base = offsets * 2
    re = tl.load(ri_ptr + base, mask=mask)
    im = tl.load(ri_ptr + base + 1, mask=mask)
    y = tl.sqrt(re * re + im * im)
    tl.store(out_ptr + offsets, y, mask=mask)


def absolute(input: torch.Tensor):
    x = input.contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if x.is_complex():
        ri = torch.view_as_real(x).contiguous()
        out_dtype = x.real.dtype
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        _absolute_complex_kernel[grid](ri, out, n_elements, BLOCK_SIZE=1024)
        return out
    else:
        out = torch.empty_like(x)
        _absolute_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
        return out


def absolute_out(input: torch.Tensor, out: torch.Tensor):
    x = input.contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if x.is_complex():
        assert (
            out.dtype == x.real.dtype
        ), "out dtype must be the real dtype of the complex input"
        assert out.shape == x.shape, "out must have the same shape as input"
        assert out.is_contiguous(), "out must be contiguous"
        ri = torch.view_as_real(x).contiguous()
        _absolute_complex_kernel[grid](ri, out, n_elements, BLOCK_SIZE=1024)
        return out
    else:
        assert out.dtype == x.dtype, "out dtype must match input dtype"
        assert out.shape == x.shape, "out must have the same shape as input"
        assert out.is_contiguous(), "out must be contiguous"
        _absolute_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
        return out
