import torch
import triton
import triton.language as tl


@triton.jit
def silu_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f = x.to(tl.float32)
    y = x_f * tl.sigmoid(x_f)
    y = y.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


_silu_kernel = silu_


def silu_(*args, **kwargs):
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))
    if x is None:
        raise ValueError("silu_ expects a tensor as the first argument (self).")
    if not x.is_cuda:
        # Fallback to PyTorch for non-CUDA tensors
        return torch.ops.aten.silu_(x)
    if not x.dtype.is_floating_point:
        raise TypeError(f"silu_ expects a floating point tensor, got {x.dtype}")
    # Fallback for unsupported dtypes or non-contiguous tensors
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if (x.dtype not in supported_dtypes) or (not x.is_contiguous()):
        return torch.ops.aten.silu_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _silu_kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
