import torch
import triton
import triton.language as tl


@triton.jit
def i0_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_f32 = x.to(tl.float32)
    ax = tl.abs(x_f32)

    # Small region: |x| <= 3.75
    t = x_f32 / 3.75
    y = t * t
    p_small = 1.0 + y * (
        3.5156229
        + y
        * (
            3.0899424
            + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
        )
    )

    # Large region: |x| > 3.75
    yb = 3.75 / ax
    p_big = 0.39894228 + yb * (
        0.01328592
        + yb
        * (
            0.00225319
            + yb
            * (
                -0.00157565
                + yb
                * (
                    0.00916281
                    + yb
                    * (
                        -0.02057706
                        + yb * (0.02635537 + yb * (-0.01647633 + yb * 0.00392377))
                    )
                )
            )
        )
    )
    # Avoid division by zero via masking; big branch only used when ax > 3.75
    res_big = tl.exp(ax) * p_big / tl.sqrt(ax)

    use_small = ax <= 3.75
    res = tl.where(use_small, p_small, res_big)

    # Store result; Triton will cast to the dtype of out_ptr as needed
    tl.store(out_ptr + offsets, res, mask=mask)


def _launch_i0(out: torch.Tensor, x: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Input and output must be CUDA tensors"
    assert (
        out.numel() == x.numel()
    ), "Input and output must have the same number of elements"
    assert out.device == x.device, "Input and output must be on the same device"

    x_in = x
    out_in = out

    # Ensure floating point compute
    if not x_in.is_floating_point():
        x_in = x_in.to(torch.get_default_dtype())

    # Cast input to match the desired output dtype if needed
    # (Compute will be done in fp32 inside kernel; store will cast to out dtype)
    if x_in.dtype != out_in.dtype:
        x_in = x_in.to(out_in.dtype)

    x_contig = x_in.contiguous()
    out_was_noncontig = not out_in.is_contiguous()
    out_contig = out_in.contiguous() if out_was_noncontig else out_in

    n_elements = out_contig.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    i0_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if out_was_noncontig:
        out_in.copy_(out_contig)
    return out_in


def i0(x: torch.Tensor):
    if not x.is_cuda:
        raise ValueError("i0: input tensor must be on CUDA device")
    # Result dtype follows PyTorch's floating type behavior; use input dtype if floating, otherwise default
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x.to(dtype=out_dtype), dtype=out_dtype, device=x.device)
    _launch_i0(out, x)
    return out


def i0_out(x: torch.Tensor, out: torch.Tensor):
    if not (x.is_cuda and out.is_cuda):
        raise ValueError("i0_out: input and output tensors must be on CUDA device")
    if not out.is_floating_point():
        raise TypeError("i0_out: output tensor must be a floating point type")
    if x.numel() != out.numel():
        raise ValueError(
            "i0_out: input and output must have the same number of elements"
        )
    _launch_i0(out, x)
    return out
