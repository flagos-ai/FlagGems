import torch
import triton
import triton.language as tl


@triton.jit
def glu_kernel(
    x_ptr,
    out_ptr,
    D_in,
    D_out,
    stride_x0,
    stride_x1,
    stride_out0,
    stride_out1,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs = tl.arange(0, BLOCK_SIZE)

    x_val_offs = pid0 * stride_x0 + pid1 * stride_x1 + offs
    x_gate_offs = pid0 * stride_x0 + (pid1 + D_out) * stride_x1 + offs

    mask = offs < D_out
    x_val = tl.load(x_ptr + x_val_offs, mask=mask, other=0.0)
    x_gate = tl.load(x_ptr + x_gate_offs, mask=mask, other=0.0)

    gate_sigmoid = tl.sigmoid(x_gate)
    acc = x_val * gate_sigmoid

    out_offs = pid0 * stride_out0 + pid1 * stride_out1 + offs
    tl.store(out_ptr + out_offs, acc, mask=mask)


def glu(x: torch.Tensor):
    """
    GLU: out = x[..., :d] * sigmoid(x[..., d:])
    input shape [*, 2*d], output shape [*, d]
    """
    assert x.shape[-1] % 2 == 0, "last dim must be even"
    D_in = x.shape[-1]
    D_out = D_in // 2
    out_shape = list(x.shape)
    out_shape[-1] = D_out
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            x.shape[0],
            triton.cdiv(D_out, meta["BLOCK_SIZE"]),
        )

    glu_kernel[grid](
        x,
        out,
        D_in,
        D_out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE=128,
    )
    return out
