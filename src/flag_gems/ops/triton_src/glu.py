import triton
import triton.language as tl
import torch


@triton.jit
def glu_kernel(
    x_ptr,
    out_ptr,
    B,
    D,
    stride_xb,
    stride_xd,
    stride_ob,
    stride_od,
    BLOCK_SIZE: tl.constexpr = 128,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < B * (D // 2)

    b_idx = offs // (D // 2)
    d_idx = offs % (D // 2)

    # a: first half, b: gate half (second half)
    a_off = b_idx * stride_xb + d_idx * stride_xd
    b_off = b_idx * stride_xb + (d_idx + D // 2) * stride_xd

    a = tl.load(x_ptr + a_off, mask=mask)
    b = tl.load(x_ptr + b_off, mask=mask)
    gate = tl.sigmoid(b)
    res = a * gate
    tl.store(out_ptr + offs, res, mask=mask)


def glu(x: torch.Tensor) -> torch.Tensor:
    """
    GLU: split last dim into a,b; return a * sigmoid(b)
    x.shape (..., D), D must even
    output shape (..., D//2)
    """
    assert x.size(-1) % 2 == 0, "GLU last dim must be even"
    out_shape = list(x.shape)
    out_shape[-1] = out_shape[-1] // 2
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    B = x.numel() // x.size(-1)
    D = x.size(-1)
    grid = lambda META: (triton.cdiv(B * (D // 2), META["BLOCK_SIZE"]),)
    glu_kernel[grid](
        x,
        out,
        B,
        D,
        x.stride(-2),
        x.stride(-1),
        out.stride(-2),
        out.stride(-1),
    )
    return out
