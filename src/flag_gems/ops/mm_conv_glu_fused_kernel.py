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
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    """
    Fused: MM + broadcast bias add + GLU split
    y = x @ w + bias
    out = y[..., :N//2] * sigmoid(y[..., N//2:])
    bias shape [N], broadcast to [M,N]
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
            mask=mask_m[:, None] & ((k + offs_k)[None, :] < K),
            other=0.0
        )
        b = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=((k + offs_k)[:, None] < K) & mask_n[None, :],
            other=0.0
        )
        acc += tl.dot(a, b)

    # broadcast bias [N]
    bias = tl.load(bias_ptr + offs_n * stride_biasn, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    half_n = N // 2
    # split last dim: a = acc[..., :half_n], gate = acc[..., half_n:]
    a_out = acc[:, :half_n]
    gate_out = acc[:, half_n:]
    glu_val = a_out * tl.sigmoid(gate_out)

    out_offs_n = offs_n[:half_n]
    out_mask_n = out_offs_n < half_n
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + out_offs_n[None, :] * stride_on,
        glu_val,
        mask=mask_m[:, None] & out_mask_n[None, :]
    )
