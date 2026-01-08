import torch
import triton
import triton.language as tl


@triton.jit
def floor_(
    x_ptr,  # pointer to input/output tensor (in-place)
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Apply floor only for floating-point dtypes; otherwise, no-op
    out = x
    if IS_FP32:
        out = tl.floor(x)
    elif IS_FP16:
        x_fp32 = tl.cast(x, tl.float32)
        out = tl.cast(tl.floor(x_fp32), tl.float16)
    elif IS_BF16:
        x_fp32 = tl.cast(x, tl.float32)
        out = tl.cast(tl.floor(x_fp32), tl.bfloat16)

    tl.store(x_ptr + offsets, out, mask=mask)


# Keep a reference to the kernel before defining the wrapper with the same name
floor__kernel = floor_


def floor_(*args, **kwargs):
    x = args[0] if len(args) > 0 else kwargs.get("input", None)
    if x is None:
        raise ValueError(
            "floor_ expects a Tensor as the first positional argument or 'input' keyword."
        )
    if not isinstance(x, torch.Tensor):
        raise TypeError("floor_ expects a torch.Tensor.")
    if not x.is_cuda:
        raise ValueError("floor_ Triton kernel requires a CUDA tensor.")
    if x.is_complex():
        raise TypeError("floor_ is not supported for complex tensors.")
    if not x.is_contiguous():
        raise ValueError(
            "floor_ Triton kernel currently supports only contiguous tensors."
        )

    n_elements = x.numel()
    if n_elements == 0:
        return x

    dtype = x.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    floor__kernel[grid](
        x,  # in-place: pass the same tensor pointer for load/store
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_FP32=IS_FP32,
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )
    return x
