import torch
import triton
import triton.language as tl


@triton.jit
def sgn_real_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Create typed constants without relying on tl.full
    zero = x - x
    one = zero + 1
    neg_one = -one

    out = tl.where(x > 0, one, tl.where(x < 0, neg_one, zero))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def sgn_complex_kernel(x_ri_ptr, out_ri_ptr, n_complex, BLOCK_SIZE: tl.constexpr):
    # x_ri_ptr and out_ri_ptr are pointers to the real-imag flattened arrays:
    # for element k: real at 2*k, imag at 2*k + 1
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_complex

    base = offsets * 2
    r = tl.load(x_ri_ptr + base, mask=mask)
    i = tl.load(x_ri_ptr + base + 1, mask=mask)

    # typed constants
    zero = r - r
    one = zero + 1

    # Compute norm and its reciprocal
    norm = tl.sqrt(r * r + i * i)
    inv = one / norm  # Will be inf when norm == 0; handled by where below

    nz = norm != 0
    out_r = tl.where(nz, r * inv, zero)
    out_i = tl.where(nz, i * inv, zero)

    tl.store(out_ri_ptr + base, out_r, mask=mask)
    tl.store(out_ri_ptr + base + 1, out_i, mask=mask)


def _sgn_impl(input: torch.Tensor) -> torch.Tensor:
    assert isinstance(input, torch.Tensor), "input must be a torch.Tensor"
    assert input.is_cuda, "input must be on CUDA device"
    # Compute into a contiguous result buffer
    result = torch.empty_like(input, memory_format=torch.contiguous_format)

    BLOCK_SIZE = 1024
    if input.is_complex():
        # Use real-imag views for complex types
        in_ri = torch.view_as_real(input).contiguous().view(-1)
        out_ri = torch.view_as_real(result).contiguous().view(-1)
        n_complex = input.numel()
        grid = lambda meta: (triton.cdiv(n_complex, meta["BLOCK_SIZE"]),)
        sgn_complex_kernel[grid](
            in_ri,
            out_ri,
            n_complex,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        x = input.contiguous().view(-1)
        out_flat = result.view(-1)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        sgn_real_kernel[grid](
            x,
            out_flat,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return result


def sgn(input: torch.Tensor, *, out: torch.Tensor = None):
    """
    Wrapper for ATen operator: ('sgn', <Autograd.disable: False>)
    """
    res = _sgn_impl(input)
    if out is not None:
        out.copy_(res)
        return out
    return res


def sgn_out(input: torch.Tensor, out: torch.Tensor):
    """
    Wrapper for ATen operator: ('sgn.out', <Autograd.disable: False>)
    """
    res = _sgn_impl(input)
    out.copy_(res)
    return out
