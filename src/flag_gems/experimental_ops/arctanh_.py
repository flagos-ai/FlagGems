import torch
import triton
import triton.language as tl


@triton.jit
def arctanh_(
    x_ptr, n_elements, BLOCK_SIZE: tl.constexpr, COMPUTE_IN_FP32: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if COMPUTE_IN_FP32:
        xf = x.to(tl.float32)
        num = 1.0 + xf
        den = 1.0 - xf
        val = 0.5 * tl.log(num / den)
        y = val.to(x.dtype)
    else:
        xf = x
        num = 1 + xf
        den = 1 - xf
        val = 0.5 * tl.log(num / den)
        y = val

    tl.store(x_ptr + offsets, y, mask=mask)


ARCTANH_KERNEL = arctanh_


def arctanh_(*args, **kwargs):
    # Extract the input tensor; accept positional or keywords like 'input' or 'self'
    x = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))
    if not isinstance(x, torch.Tensor):
        raise TypeError("arctanh_ expects a single Tensor argument")

    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    if not x.is_floating_point():
        raise TypeError("arctanh_ only supports floating point tensors")

    n_elements = x.numel()
    if n_elements == 0:
        return x

    use_fp32 = x.dtype in (torch.float16, torch.bfloat16)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    ARCTANH_KERNEL[grid](x, n_elements, BLOCK_SIZE=1024, COMPUTE_IN_FP32=use_fp32)
    return x
