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
    acc_a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, K, BLOCK_K):
        offs_k_cur = k + offs_k
        mask_k = offs_k_cur < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k_cur[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )
        wa = tl.load(
            wa_ptr + offs_k_cur[:, None] * stride_wak + offs_n[None, :] * stride_wan,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0
        )
        wg = tl.load(
            wg_ptr + offs_k_cur[:, None] * stride_wgk + offs_n[None, :] * stride_wgn,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0
        )
        acc_a += tl.dot(x, wa)
        acc_g += tl.dot(x, wg)

    # bias add
    bias_a = tl.load(ba_ptr + offs_n * stride_ban, mask=mask_n, other=0.0)
    bias_g = tl.load(bg_ptr + offs_n * stride_bgn, mask=mask_n, other=0.0)
    acc_a += bias_a[None, :]
    acc_g += bias_g[None, :]

    # GLU: a * sigmoid(b) 【审核重点修复】
    glu_val = acc_a * tl.sigmoid(acc_g)
    glu_val = glu_val.to(tl.float16)

    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        glu_val, mask=mask_out
    )
