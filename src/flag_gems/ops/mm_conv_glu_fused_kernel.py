import triton
import triton.language as tl

@triton.jit
def mm_conv_glu_fused_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_biasn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 32,  # 注意：这里N是输出维度=N_orig//2
    BLOCK_K: tl.constexpr = 64,
):
    """
    Fused MM + broadcast bias + GLU for Iluvatar CoreX
    Equivalent:
        y = x @ w_full + bias  # w_full [K, 2*N], bias [2*N]
        out = y[..., :N] * sigmoid(y[..., N:])
    Adaptation: split weight into w_a, w_gate, bias into bias_a, bias_gate
    No dynamic tensor slice (Iluvatar triton limitation)
    N in args = original_N // 2
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc_a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
            mask=mask_m[:, None] & ((k + offs_k)[None, :] < K),
            other=0.0
        )
        w_a = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=((k + offs_k)[:, None] < K) & mask_n[None, :],
            other=0.0
        )
        w_gate = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + (offs_n + N)[None, :] * stride_wn,
            mask=((k + offs_k)[:, None] < K) & mask_n[None, :],
            other=0.0
        )
        acc_a += tl.dot(x, w_a)
        acc_gate += tl.dot(x, w_gate)

    # broadcast bias load
    bias_a = tl.load(bias_ptr + offs_n * stride_biasn, mask=mask_n, other=0.0)
    bias_gate = tl.load(bias_ptr + (offs_n + N) * stride_biasn, mask=mask_n, other=0.0)
    acc_a = acc_a + bias_a[None, :]
    acc_gate = acc_gate + bias_gate[None, :]

    glu_val = acc_a * tl.sigmoid(acc_gate)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        glu_val,
        mask=mask_m[:, None] & mask_n[None, :]
    )
