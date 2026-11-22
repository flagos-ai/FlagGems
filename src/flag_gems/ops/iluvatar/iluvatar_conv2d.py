import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_fwd_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, H, W, C_in, C_out,
    K_h, K_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    OH, OW,
    BLOCK_N: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_idx = pid // ((OH * OW + BLOCK_OH * BLOCK_OW - 1) // (BLOCK_OH * BLOCK_OW))
    rem = pid % ((OH * OW + BLOCK_OH * BLOCK_OW - 1) // (BLOCK_OH * BLOCK_OW))
    oh_start = (rem // ((OW + BLOCK_OW - 1) // BLOCK_OW)) * BLOCK_OH
    ow_start = (rem % ((OW + BLOCK_OW - 1) // BLOCK_OW)) * BLOCK_OW
    cout_start = tl.program_id(axis=1) * BLOCK_COUT

    offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oh = oh_start + tl.arange(0, BLOCK_OH)
    offs_ow = ow_start + tl.arange(0, BLOCK_OW)
    offs_cout = cout_start + tl.arange(0, BLOCK_COUT)

    mask_n = offs_n < N
    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_cout = offs_cout < C_out

    out = tl.zeros((BLOCK_N, BLOCK_OH, BLOCK_OW, BLOCK_COUT), dtype=tl.float32)

    for kh in range(K_h):
        for kw in range(K_w):
            for cin in range(C_in):
                ih = offs_oh * stride_h + kh * dilation_h - pad_h
                iw = offs_ow * stride_w + kw * dilation_w - pad_w
                mask_ih = (ih >= 0) & (ih < H)
                mask_iw = (iw >= 0) & (iw < W)
                x_mask = mask_n[:, None, None, None] & mask_oh[None, :, None, None] & mask_ow[None, None, :, None] & mask_ih[None, :, None, None] & mask_iw[None, None, :, None]

                x_off = offs_n[:, None, None, None] * H * W * C_in + ih[None, :, None, None] * W * C_in + iw[None, None, :, None] * C_in + cin
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                w_off = offs_cout[None, None, None, :] * C_in * K_h * K_w + cin * K_h * K_w + kh * K_w + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_cout[None, None, None, :], other=0.0)

                out += x_val * w_val

    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + offs_cout, mask=mask_cout, other=0.0)
        out += bias_val[None, None, None, :]

    out_mask = mask_n[:, None, None, None] & mask_oh[None, :, None, None] & mask_ow[None, None, :, None] & mask_cout[None, None, None, :]
    out_off = offs_n[:, None, None, None] * OH * OW * C_out + offs_oh[None, :, None, None] * OW * C_out + offs_ow[None, None, :, None] * C_out + offs_cout[None, None, None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)

def iluvatar_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int = 1,
) -> torch.Tensor:
    assert groups == 1, "only support groups=1 for now"
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    OH = (H + 2 * pad_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dilation_w * (K_w - 1) - 1) // stride_w + 1

    out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_N = 2
    BLOCK_OH = 8
    BLOCK_OW = 8
    BLOCK_COUT = 16

    grid = (
        triton.cdiv(N, BLOCK_N) * triton.cdiv(OH, BLOCK_OH) * triton.cdiv(OW, BLOCK_OW),
        triton.cdiv(C_out, BLOCK_COUT),
    )

    _conv2d_fwd_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        bias_ptr=bias if bias is not None else 0,
        out_ptr=out,
        N=N, H=H, W=W, C_in=C_in, C_out=C_out,
        K_h=K_h, K_w=K_w,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        OH=OH, OW=OW,
        BLOCK_N=BLOCK_N,
        BLOCK_OH=BLOCK_OH,
        BLOCK_OW=BLOCK_OW,
        BLOCK_COUT=BLOCK_COUT,
    )
    return out

if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn((2, 16, 13, 13), device="cuda", dtype=torch.float16)
    w = torch.randn((32, 16, 3, 3), device="cuda", dtype=torch.float16)
    b = torch.randn((32,), device="cuda", dtype=torch.float16)
    stride = (2, 2)
    pad = (1, 1)
    dil = (1, 1)

    ref = torch.nn.functional.conv2d(x, w, b, stride, pad, dil, groups=1)
    tri = iluvatar_conv2d(x, w, b, stride, pad, dil)
    print("max diff:", (ref - tri).abs().max())
