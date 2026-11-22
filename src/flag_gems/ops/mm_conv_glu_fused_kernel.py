import triton
import triton.language as tl

@triton.jit
def mm_conv_glu_fused_kernel(
    x_ptr,
    wa_ptr,
    wg_ptr,
    ba_ptr,
    bg_ptr,
    out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wak, stride_wan,
    stride_wgk, stride_wgn,
    stride_ban, stride_bgn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wa_ptrs = wa_ptr + offs_k[:, None] * stride_wak + offs_n[None, :] * stride_wan
    wg_ptrs = wg_ptr + offs_k[:, None] * stride_wgk + offs_n[None, :] * stride_wgn

    acc_a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K
        mask_k = k_offs + offs_k < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        wa = tl.load(wa_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        wg = tl.load(wg_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_a += tl.dot(x, wa)
        acc_g += tl.dot(x, wg)
        x_ptrs += BLOCK_K * stride_xk
        wa_ptrs += BLOCK_K * stride_wak
        wg_ptrs += BLOCK_K * stride_wgk

    ba = tl.load(ba_ptr + offs_n * stride_ban, mask=offs_n < N, other=0.0)
    bg = tl.load(bg_ptr + offs_n * stride_bgn, mask=offs_n < N, other=0.0)
    y_a = acc_a + ba[None, :]
    y_g = acc_g + bg[None, :]
    out_val = y_a * tl.sigmoid(y_g)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out_val, mask=mask_m[:, None] & mask_n[None, :])
