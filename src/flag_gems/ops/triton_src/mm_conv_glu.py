import torch
import triton
import triton.language as tl


@triton.jit
def mm_conv_glu_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    Cout,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wc,
    stride_om,
    stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_c = offs_c < Cout

    acc_gate = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    acc_val = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < K

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        # w1: gate 前半段
        w_gate = tl.load(
            w_ptr + k_offs[:, None] * stride_wk + offs_c[None, :] * stride_wc,
            mask=mask_k[:, None] & mask_c[None, :],
            other=0.0,
        )
        # w2: val 后半段
        w_val = tl.load(
            w_ptr + k_offs[:, None] * stride_wk + (offs_c[None, :] + Cout) * stride_wc,
            mask=mask_k[:, None] & mask_c[None, :],
            other=0.0,
        )
        acc_gate += tl.dot(a, w_gate)
        acc_val += tl.dot(a, w_val)

    # 和测试用例对齐：(a@w1) * sigmoid(a@w2)
    out = acc_gate * tl.sigmoid(acc_val)
    out_fp16 = out.to(tl.float16)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_c[None, :] * stride_oc,
        out_fp16,
        mask=mask_m[:, None] & mask_c[None, :],
    )


def mm_conv_glu(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    GLU after matmul: out = (a @ w1) * sigmoid(a @ w2)
    a: [M, K]
    w: [K, 2*Cout]  -- 前半Cout是w1(gate)，后半Cout是w2(val)
    与 test_mm_conv_glu_fused 参考公式保持一致
    return: [M, Cout]
    """
    assert a.is_cuda and w.is_cuda
    assert a.dtype == torch.float16 and w.dtype == torch.float16
    M, K = a.shape
    K2, Cout2 = w.shape
    assert K == K2
    assert Cout2 % 2 == 0
    Cout = Cout2 // 2
    out = torch.empty((M, Cout), device=a.device, dtype=a.dtype)

    BLOCK_M = 64
    BLOCK_C = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Cout, BLOCK_C))
    mm_conv_glu_kernel[grid](
        a,
        w,
        out,
        M,
        Cout,
        K,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
    )
    return out
