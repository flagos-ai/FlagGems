import torch
import triton
import triton.language as tl


@triton.jit
def sign_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    zero = x - x
    one = zero + 1
    minus_one = zero - 1

    pos = x > 0
    neg = x < 0

    res = tl.where(pos, one, zero)
    res = tl.where(neg, minus_one, res)

    # Preserve NaN for floating-point inputs
    is_nan = x != x
    res = tl.where(is_nan, x, res)

    tl.store(x_ptr + offsets, res, mask=mask)


_sign_kernel = sign_


def sign_(*args, **kwargs):
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", kwargs.get("x", None)))
    if x is None:
        raise ValueError("sign_: expected a Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("sign_: expected a torch.Tensor as input")

    if x.numel() == 0:
        return x

    supported_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )

    if (not x.is_cuda) or (x.dtype not in supported_dtypes):
        torch.sign_(x)
        return x

    BLOCK_SIZE = 1024

    if not x.is_contiguous():
        tmp = x.contiguous()
        n_elements = tmp.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _sign_kernel[grid](tmp, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x.copy_(tmp)
        return x

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _sign_kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
