import torch
import triton
import triton.language as tl


@triton.jit
def arccosh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x32 = x.to(tl.float32)

    # acosh(x) = log(x + sqrt(x - 1) * sqrt(x + 1))
    s1 = tl.sqrt(x32 - 1.0)
    s2 = tl.sqrt(x32 + 1.0)
    y32 = tl.log(x32 + s1 * s2)

    tl.store(out_ptr + offsets, y32, mask=mask)


def arccosh(input: torch.Tensor):
    assert input.is_cuda, "Input tensor must be on CUDA device"
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "Supported dtypes: float16, bfloat16, float32"

    x = input.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    arccosh_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def arccosh_out(input: torch.Tensor, out: torch.Tensor):
    assert input.is_cuda and out.is_cuda, "Tensors must be on CUDA device"
    assert input.shape == out.shape, "Input and out must have the same shape"
    assert input.dtype == out.dtype, "Input and out must have the same dtype"
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "Supported dtypes: float16, bfloat16, float32"

    x = input.contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if out.is_contiguous():
        arccosh_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    else:
        tmp = torch.empty_like(x)
        arccosh_kernel[grid](x, tmp, n_elements, BLOCK_SIZE=1024)
        out.copy_(tmp)
    return out
