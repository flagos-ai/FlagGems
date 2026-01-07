import torch
import triton
import triton.language as tl


@triton.jit
def _special_i0e_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute in fp32 for accuracy/stability
    xf = x.to(tl.float32)
    ax = tl.abs(xf)

    # Small region: x <= 3.75
    t_small = ax / 3.75
    t2 = t_small * t_small
    # Polynomial approximation for I0 in small region (Numerical Recipes)
    p = 1.0 + t2 * (
        3.5156229
        + t2
        * (
            3.0899424
            + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))
        )
    )
    small = p * tl.exp(-ax)

    # Large region: x > 3.75, use asymptotic expansion to avoid exp overflow
    # i0e(x) = I0(x)*exp(-|x|) ≈ (1/sqrt(|x|)) * poly(3.75/|x|)
    t = 3.75 / ax
    q = 0.39894228 + t * (
        0.01328592
        + t
        * (
            0.00225319
            + t
            * (
                -0.00157565
                + t
                * (
                    0.00916281
                    + t
                    * (
                        -0.02057706
                        + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))
                    )
                )
            )
        )
    )
    large = q / tl.sqrt(ax)

    is_large = ax > 3.75
    y = tl.where(is_large, large, small)

    # Cast back to input dtype for storage
    y = y.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _run_special_i0e_kernel(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Tensors must be CUDA tensors"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "Unsupported dtype"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

    x_c = x.contiguous()
    out_c = out.contiguous()

    n_elements = out_c.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _special_i0e_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_i0e(x: torch.Tensor):
    """
    ATen wrapper: special_i0e(Tensor self) -> Tensor
    """
    out = torch.empty_like(x)
    return _run_special_i0e_kernel(x, out)


def special_i0e_out(x: torch.Tensor, out: torch.Tensor):
    """
    ATen wrapper: special_i0e.out(Tensor self, Tensor out) -> Tensor
    """
    # Broadcast input to out's shape if needed
    if x.shape != out.shape:
        x = x.expand(out.shape)
    _run_special_i0e_kernel(x, out)
    return out
