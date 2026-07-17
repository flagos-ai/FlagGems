import torch
import triton
import triton.language as tl


@triton.jit
def rsqrt_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    res_fp32 = 1.0 / tl.sqrt(x_fp32)
    res = res_fp32.to(x.dtype)
    tl.store(x_ptr + offsets, res, mask=mask)


# Keep a handle to the Triton kernel before defining the Python wrapper with the same name.
rsqrt__triton_kernel = rsqrt_


def rsqrt_(*args, **kwargs):
    # Resolve input tensor from positional or keyword arguments
    x = None
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", None)
        if x is None:
            x = kwargs.get("self", None)

    if x is None:
        raise ValueError("rsqrt_ expects a tensor as its first argument")

    if not isinstance(x, torch.Tensor):
        raise TypeError("rsqrt_ expects a torch.Tensor")

    if not x.is_cuda:
        raise AssertionError("Input tensor must be on CUDA device")

    if not x.is_contiguous():
        raise AssertionError("Input tensor must be contiguous")

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(
            "rsqrt_ only supports floating point tensors (float16, bfloat16, float32, float64)"
        )

    n_elements = x.numel()
    if n_elements == 0:
        return x

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    rsqrt__triton_kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
