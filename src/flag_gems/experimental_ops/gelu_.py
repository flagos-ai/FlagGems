import torch
import triton
import triton.language as tl


@triton.jit
def gelu_(
    x_ptr,  # *Pointer* to the input/output tensor (in-place).
    n_elements,  # Number of elements.
    USE_TANH: tl.constexpr,  # Whether to use tanh approximation.
    BLOCK_SIZE: tl.constexpr,  # Elements per program.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_f32 = x.to(tl.float32)

    # Compute GELU either exact (via erf approximation) or tanh approximation
    if USE_TANH:
        # tanh approximation:
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        x3 = x_f32 * x_f32 * x_f32
        z = c0 * (x_f32 + c1 * x3)
        # tanh(z) = (1 - e^{-2z}) / (1 + e^{-2z})
        t = tl.exp(-2.0 * z)
        tanh_z = (1.0 - t) / (1.0 + t)
        y = 0.5 * x_f32 * (1.0 + tanh_z)
    else:
        # exact (erf-based) GELU:
        # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        inv_sqrt2 = 0.7071067811865476
        z = x_f32 * inv_sqrt2

        # Abramowitz and Stegun formula 7.1.26 for erf approximation
        # erf(x) ≈ sign(x) * (1 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * e^{-x^2})
        # where t = 1 / (1 + p*|x|)
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        az = tl.abs(z)
        t = 1.0 / (1.0 + p * az)
        poly = a5
        poly = poly * t + a4
        poly = poly * t + a3
        poly = poly * t + a2
        poly = poly * t + a1
        poly = poly * t
        erf_abs = 1.0 - poly * tl.exp(-az * az)
        erf_z = tl.where(z >= 0, erf_abs, -erf_abs)

        y = 0.5 * x_f32 * (1.0 + erf_z)

    y_cast = y.to(x.dtype)
    tl.store(x_ptr + offsets, y_cast, mask=mask)


# Preserve a handle to the kernel before defining the Python wrapper of the same name
gelu__kernel = gelu_


def gelu_(*args, **kwargs):
    # Resolve input tensor
    x = None
    if len(args) >= 1:
        x = args[0]
    else:
        # Try common names
        x = kwargs.get("input", None)
        if x is None:
            x = kwargs.get("self", None)
        if x is None:
            x = kwargs.get("x", None)
    if x is None:
        raise ValueError("gelu_ expects a tensor as the first argument.")

    # Determine approximation mode
    approx = kwargs.get("approximate", "none")
    if isinstance(approx, bool):
        use_tanh = bool(approx)
    else:
        approx_str = str(approx).lower()
        if approx_str in ("tanh", "true"):
            use_tanh = True
        elif approx_str in ("none", "false"):
            use_tanh = False
        else:
            raise ValueError(
                f"Unsupported approximate mode: {approx}. Use 'none' or 'tanh'."
            )

    if not x.is_cuda:
        raise AssertionError("Input tensor must be on CUDA device for Triton kernel.")
    if not x.is_contiguous():
        raise AssertionError("Input tensor must be contiguous.")
    if not x.is_floating_point():
        raise AssertionError("gelu_ expects a floating point tensor.")

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    gelu__kernel[grid](x, n_elements, USE_TANH=use_tanh, BLOCK_SIZE=BLOCK_SIZE)
    return x
