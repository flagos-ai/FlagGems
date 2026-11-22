import torch
import triton
import triton.language as tl

@triton.jit
def _mm_conv_glu_kernel(
    a_ptr, w_ptr, out_ptr,
    M, Cout_full, K,
    stride_am, stride_ak,
    stride_wk, stride_wc,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N_HALF: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_half = pid_n * BLOCK_N_HALF + tl.arange(0, BLOCK_N_HALF)
    offs_k = tl.arange(0, BLOCK_K)

    # val 位置：0 ~ half-1
    val_offs = offs_n_half
    # gate 位置：half ~ Cout_full
    gate_offs = offs_n_half + (Cout_full // 2)

    acc_val = tl.zeros((BLOCK_M, BLOCK_N_HALF), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N_HALF), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0
        )
        w_val = tl.load(
            w_ptr + (k + offs_k)[:, None] * stride_wk + val_offs[None, :] * stride_wc,
            mask=((k + offs_k)[:, None] < K) & (val_offs[None, :] < Cout_full//2),
            other=0.0
        )
        w_gate = tl.load(
            w_ptr + (k + offs_k)[:, None] * stride_wk + gate_offs[None, :] * stride_wc,
            mask=((k + offs_k)[:, None] < K) & (gate_offs[None, :] < Cout_full),
            other=0.0
        )
        acc_val += tl.dot(a, w_val)
        acc_gate += tl.dot(a, w_gate)

    glu_out = acc_val * tl.sigmoid(acc_gate)
    store_mask = (offs_m[:, None] < M) & (offs_n_half[None, :] < (Cout_full//2))
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n_half[None, :] * stride_on,
        glu_out.to(tl.float16),
        mask=store_mask
    )

def iluvatar_mm_conv_glu(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # x: [M, K], w: [K, Cout_full]
    # GLU: out = (A@W[:,0:half]) * sigmoid(A@W[:,half:])
    # output shape: [M, Cout_full//2]
    assert x.shape[1] == w.shape[0]
    M, K = x.shape
    K_w, Cout_full = w.shape
    assert Cout_full % 2 == 0, "GLU require last dim even"
    Cout = Cout_full // 2
    out = torch.empty((M, Cout), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N_HALF = 32
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Cout, BLOCK_N_HALF))

    _mm_conv_glu_kernel[grid](
        x, w, out,
        M, Cout_full, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N_HALF=BLOCK_N_HALF,
        BLOCK_K=BLOCK_K
    )
    return out
