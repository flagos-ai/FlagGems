import torch
import triton
import triton.language as tl


@triton.jit
def i0_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = tl.cast(x, tl.float32)
    ax = tl.abs(xf)

    t_small = ax / 3.75
    y_small = t_small * t_small
    poly_small = 1.0 + y_small * (
        3.5156229 + y_small * (
            3.0899424 + y_small * (
                1.2067492 + y_small * (
                    0.2659732 + y_small * (
                        0.0360768 + y_small * 0.0045813
                    )
                )
            )
        )
    )

    y_large = 3.75 / ax
    poly_large = 0.39894228 + y_large * (
        0.01328592 + y_large * (
            0.00225319 + y_large * (
                -0.00157565 + y_large * (
                    0.00916281 + y_large * (
                        -0.02057706 + y_large * (
                            0.02635537 + y_large * (
                                -0.01647633 + y_large * 0.00392377
                            )
                        )
                    )
                )
            )
        )
    )
    val_large = tl.exp(ax) * poly_large / tl.sqrt(ax)

    result = tl.where(ax <= 3.75, poly_small, val_large)

    result_cast = tl.cast(result, x.dtype)
    tl.store(x_ptr + offsets, result_cast, mask=mask)


# Keep a reference to the Triton kernel before defining the Python wrapper with the same name
i0__kernel = i0_


def i0_(*args, **kwargs):
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        # Try common keyword names
        for k in ("input", "self", "x"):
            if k in kwargs:
                x = kwargs[k]
                break
    if x is None:
        raise ValueError("i0_ expects a tensor as the first positional argument or in keyword 'input'/'self'/'x'.")

    if not x.is_cuda:
        raise AssertionError("Input tensor must be on a CUDA device.")
    if not x.is_contiguous():
        raise AssertionError("Input tensor must be contiguous.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise AssertionError("Unsupported dtype for i0_. Supported: float16, bfloat16, float32, float64.")

    n_elements = x.numel()
    if n_elements == 0:
        return x

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    i0__kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x