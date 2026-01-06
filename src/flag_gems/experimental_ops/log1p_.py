import torch
import triton
import triton.language as tl


@triton.jit
def log1p_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    y = tl.log(x_fp32 + 1.0)
    y_cast = y.to(x.dtype)

    tl.store(x_ptr + offsets, y_cast, mask=mask)


_log1p_kernel = log1p_


def log1p_(*args, **kwargs):
    # Accept tensor from positional or keyword args
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("input", None)

    if x is None:
        raise ValueError(
            "log1p_ expects a tensor as the first argument or keyword 'input'."
        )

    if not isinstance(x, torch.Tensor):
        raise TypeError("log1p_ expects a torch.Tensor as input.")

    # Fallback for unsupported device/dtype/layout
    if not x.is_cuda:
        return torch.ops.aten.log1p_(x)
    if not x.is_contiguous():
        return torch.ops.aten.log1p_(x)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.ops.aten.log1p_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _log1p_kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
