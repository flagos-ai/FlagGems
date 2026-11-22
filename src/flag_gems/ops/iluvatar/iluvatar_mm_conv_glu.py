import torch
import triton
import triton.language as tl

@triton.jit
def _mm_conv_glu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, Cout,
    stride_xm, stride_xk,
    stride_wk, stride_wc,
    stride_bc,
    stride_om, stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_c = offs_c < Cout
    k_mask = offs_k < K

    acc_a = tl.zeros((BLOCK_M, BLOCK_COUT), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_COUT), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K
        x_offs = offs_m[:, None] * stride_xm + (k_offs + offs_k)[None, :] * stride_xk
        x = tl.load(x_ptr + x_offs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        # 原w是K, 2*Cout；前半通道a: [0:Cout]，后半gate:[Cout:2Cout]，不使用::2
        w_a_offs = (k_offs + offs_k)[:, None] * stride_wk + offs_c[None, :] * stride_wc
        w_g_offs = (k_offs + offs_k)[:, None] * stride_wk + (offs_c + Cout)[None, :] * stride_wc

        w_a = tl.load(w_ptr + w_a_offs, mask=k_mask[:, None] & mask_c[None, :], other=0.0)
        w_g = tl.load(w_ptr + w_g_offs, mask=k_mask[:, None] & mask_c[None, :], other=0.0)

        acc_a += tl.dot(x, w_a)
        acc_gate += tl.dot(x, w_g)

    # bias
    bias_a = tl.load(b_ptr + offs_c * stride_bc, mask=mask_c, other=0.0)
    bias_g = tl.load(b_ptr + (offs_c + Cout) * stride_bc, mask=mask_c, other=0.0)
    acc_a = acc_a + bias_a[None, :]
    acc_gate = acc_gate + bias_g[None, :]

    # GLU 公式：a * sigmoid(gate)
    glu_val = acc_a * tl.sigmoid(acc_gate)
    out = glu_val.to(tl.float16)
    out_mask = mask_m[:, None] & mask_c[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_c[None, :] * stride_oc, out, mask=out_mask)

def iluvatar_mm_conv_glu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_COUT=64, BLOCK_K=64):
    M, K = x.shape
    Kw, Cout2 = w.shape
    assert Kw == K
    Cout = Cout2 // 2
    # 输出shape是 [M, Cout]，不是 [M, Cout2]，修复torch.empty_like错误
    out = torch.empty((M, Cout), device=x.device, dtype=x.dtype)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_c = (Cout + BLOCK_COUT - 1) // BLOCK_COUT
    grid = (grid_m, grid_c)

    _mm_conv_glu_kernel[grid](
        x, w, b, out,
        M, K, Cout,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_COUT=BLOCK_COUT, BLOCK_K=BLOCK_K
    )
    return out
