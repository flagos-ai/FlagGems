import torch
import triton
import triton.language as tl


@triton.jit
def sinh_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y = 0.5 * (tl.exp(x_f32) - tl.exp(-x_f32))
    y_cast = y.to(x.dtype)
    tl.store(x_ptr + offsets, y_cast, mask=mask)


# Keep a reference to the Triton kernel before defining the Python wrapper with the same name.
_sinh_kernel = sinh_


def sinh_(*args, **kwargs):
    # Accept various calling conventions: sinh_(tensor), sinh_(self=tensor), sinh_(input=tensor)
    x = None
    if args:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input", None))
    if x is None:
        raise TypeError("sinh_ expected a Tensor as the first argument")

    if not isinstance(x, torch.Tensor):
        raise TypeError("sinh_ expected a torch.Tensor")

    if x.numel() == 0:
        return x

    if not x.is_cuda:
        raise RuntimeError("sinh_ Triton kernel requires a CUDA tensor")

    if not x.is_contiguous():
        raise RuntimeError(
            "sinh_ Triton kernel currently supports only contiguous tensors"
        )

    supported_dtypes = (torch.float16, torch.float32, torch.bfloat16)
    if x.dtype not in supported_dtypes:
        raise RuntimeError(
            f"sinh_ Triton kernel supports dtypes {supported_dtypes}, but got {x.dtype}"
        )

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _sinh_kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
