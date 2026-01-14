import torch
import triton
import triton.language as tl


@triton.jit
def erf_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x32 = x.to(tl.float32)

    ax = tl.abs(x32)
    t = 1.0 / (1.0 + 0.5 * ax)

    p = 1.00002368 + t * (
        0.37409196
        + t
        * (
            0.09678418
            + t
            * (
                -0.18628806
                + t
                * (
                    0.27886807
                    + t
                    * (
                        -1.13520398
                        + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
                    )
                )
            )
        )
    )
    s = -x32 * x32 - 1.26551223 + t * p
    tau = t * tl.exp(s)
    y32 = tl.where(x32 >= 0, 1.0 - tau, tau - 1.0)

    y = y32.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


# keep a reference to the kernel before defining the wrapper with the same name
erf__kernel = erf_


def erf_(*args, **kwargs):
    # Extract the input tensor
    x = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
        x = kwargs["input"]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        x = kwargs["self"]
    elif (
        "args" in kwargs
        and isinstance(kwargs["args"], (list, tuple))
        and kwargs["args"]
    ):
        x = kwargs["args"][0]
    if x is None:
        raise TypeError("erf_ expects a tensor as its first argument")

    # Fallback for unsupported devices/dtypes
    if not x.is_cuda:
        return x.erf_()

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return x.erf_()

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    erf__kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
