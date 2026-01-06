import torch
import triton
import triton.language as tl


@triton.jit
def erfinv_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and compute in fp32 for better precision
    y = tl.load(x_ptr + offsets, mask=mask, other=0)
    y32 = tl.cast(y, tl.float32)

    # Constants
    a = 0.147
    inv_a = 1.0 / a
    PI = 3.14159265358979323846
    SQRT_PI = 1.77245385090551602729

    # Winitzki approximation for initial guess
    # w = ln(1 - y^2)
    w = tl.log(1.0 - y32 * y32)
    b = (2.0 / (PI * a)) + 0.5 * w
    # inner = sqrt(b^2 - w/a)
    inner = tl.sqrt(b * b - w * inv_a)
    # sign(y)
    s = tl.where(y32 >= 0.0, 1.0, -1.0)
    x0 = s * tl.sqrt(inner - b)

    # Valid mask for refinement: -1 < y < 1
    valid = (y32 > -1.0) & (y32 < 1.0)

    # Newton refinement using an erf approximation (Abramowitz-Stegun 7.1.26)
    # Perform two iterations
    for _ in range(2):
        z = x0

        absz = tl.abs(z)
        t = 1.0 / (1.0 + 0.3275911 * absz)
        # Polynomial for approximation
        poly = (
            ((((1.061405429 * t) - 1.453152027) * t) + 1.421413741) * t - 0.284496736
        ) * t + 0.254829592
        erf_abs = 1.0 - poly * t * tl.exp(-absz * absz)
        erf_z = tl.where(z >= 0.0, erf_abs, -erf_abs)

        derivative = (2.0 / SQRT_PI) * tl.exp(-(z * z))
        step = (erf_z - y32) / derivative
        x0 = tl.where(valid, z - step, z)

    # Store result back in-place
    tl.store(x_ptr + offsets, x0, mask=mask)


_erfinv_kernel = erfinv_


def erfinv_(*args, **kwargs):
    # Expect a single tensor input
    x = args[0] if len(args) > 0 else kwargs.get("input", None)
    if x is None:
        raise ValueError("erfinv_ expects a tensor as the first argument.")
    # Fallback for unsupported cases
    if (not x.is_cuda) or (x.dtype == torch.float64) or (not x.is_contiguous()):
        x.copy_(torch.special.erfinv(x))
        return x

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "erfinv_ Triton kernel supports float16, bfloat16, and float32 tensors."
        )

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _erfinv_kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
