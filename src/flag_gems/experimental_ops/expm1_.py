import torch
import triton
import triton.language as tl


@triton.jit
def expm1_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    y = tl.exp(x_fp32) - 1.0
    y = y.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


# Preserve a handle to the Triton kernel before defining the Python wrapper with the same name.
expm1___kernel = expm1_


def expm1_(*args, **kwargs):
    # Extract the input tensor from args/kwargs
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        for key in ("input", "self", "x", "tensor"):
            if key in kwargs:
                x = kwargs[key]
                break
    if x is None:
        raise TypeError("expm1_ expected a tensor as the first argument")

    if not isinstance(x, torch.Tensor):
        raise TypeError("expm1_ expected a torch.Tensor")

    if not x.is_cuda:
        raise AssertionError("expm1_ Triton kernel requires a CUDA tensor")

    if not x.is_floating_point():
        raise TypeError("expm1_ only supports floating point tensors")

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if x.is_contiguous():
        expm1___kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # Fallback: operate on a contiguous copy, then copy back in-place
        tmp = x.contiguous()
        expm1___kernel[grid](tmp, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x.copy_(tmp)

    return x
