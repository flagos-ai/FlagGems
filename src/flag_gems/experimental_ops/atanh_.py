import torch
import triton
import triton.language as tl


@triton.jit
def atanh_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_fp32 = x.to(tl.float32)
    numerator = 1.0 + x_fp32
    denominator = 1.0 - x_fp32
    out_fp32 = 0.5 * tl.log(numerator / denominator)
    out = out_fp32.to(x.dtype)

    tl.store(x_ptr + offsets, out, mask=mask)


# Preserve a handle to the Triton kernel before defining the Python wrapper of the same name
atanh__triton_kernel = atanh_


def atanh_(*args, **kwargs):
    if len(args) < 1:
        raise TypeError("atanh_ expects at least one argument: a torch.Tensor")
    x = args[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError("atanh_ expects a torch.Tensor as the first argument")
    if not x.is_cuda:
        raise ValueError("atanh_ expects the tensor to be on a CUDA device")
    if not x.is_floating_point():
        raise TypeError("atanh_ expects a floating-point tensor")

    # Work on a contiguous buffer, then copy results back to x to preserve in-place semantics.
    xc = x.contiguous()
    n_elements = xc.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    atanh__triton_kernel[grid](xc, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    x.copy_(xc)
    return x
