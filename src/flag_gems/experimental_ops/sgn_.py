import torch
import triton
import triton.language as tl


@triton.jit
def sgn_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    pos = x > 0
    neg = x < 0
    res = pos.to(x.dtype) - neg.to(x.dtype)

    # Propagate NaNs for floating types. For integer types, (x != x) is always false.
    is_nan = x != x
    res = tl.where(is_nan, x, res)

    tl.store(x_ptr + offsets, res, mask=mask)


sgn___kernel = sgn_


def sgn_(*args, **kwargs):
    # Expect a single tensor argument (in-place op)
    x = None
    if len(args) == 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
        x = kwargs["input"]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        x = kwargs["self"]

    if x is None:
        raise TypeError("sgn_ expects a single Tensor argument")

    # Fallback for unsupported cases
    unsupported = (
        (not x.is_cuda)
        or (not x.is_contiguous())
        or x.is_complex()
    )
    supported_dtypes = {
        torch.float16, torch.float32, torch.float64, torch.bfloat16,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    }
    if unsupported or x.dtype not in supported_dtypes:
        return torch.ops.aten.sgn_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    sgn___kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x