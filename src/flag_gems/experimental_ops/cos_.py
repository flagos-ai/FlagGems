import torch
import triton
import triton.language as tl


@triton.jit
def cos_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    y = tl.cos(x_fp32)
    y = y.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


# Preserve reference to the kernel before defining the wrapper with the same name.
cos__kernel = cos_


def cos_(*args, **kwargs):
    # Expect a single tensor input, similar to torch.ops.aten.cos_
    x = None
    if len(args) == 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
        x = kwargs["input"]
    else:
        raise TypeError(
            "cos_ expects a single Tensor argument (positional or keyword 'input')."
        )

    # Fallback to PyTorch for unsupported cases
    if (
        (not x.is_cuda)
        or (not x.is_contiguous())
        or (
            x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        )
    ):
        return torch.cos_(x)

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    cos__kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
