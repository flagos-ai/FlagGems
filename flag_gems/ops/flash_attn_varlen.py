import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_varlen_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    max_seqlen_q,
    max_seqlen_k,
    q_h_stride,
    q_s_stride,
    q_d_stride,
    k_h_stride,
    k_s_stride,
    k_d_stride,
    v_h_stride,
    v_s_stride,
    v_d_stride,
    o_h_stride,
    o_s_stride,
    o_d_stride,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    cu_q_start = tl.load(cu_seqlens_q_ptr + pid)
    cu_q_end = tl.load(cu_seqlens_q_ptr + pid + 1)
    seq_q_len = cu_q_end - cu_q_start

    q_base = q_ptr + head_idx * q_h_stride + cu_q_start * q_s_stride
    q = tl.load(
        q_base + offs_m[:, None] * q_s_stride + offs_d[None, :] * q_d_stride,
        mask=offs_m[:, None] < seq_q_len,
        other=0.0,
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_k in range(0, max_seqlen_k, BLOCK_N):
        offs_k = start_k + offs_n
        cu_k_start = tl.load(cu_seqlens_k_ptr + pid)
        cu_k_end = tl.load(cu_seqlens_k_ptr + pid + 1)
        seq_k_len = cu_k_end - cu_k_start
        k_mask = offs_k < seq_k_len

        k_base = k_ptr + head_idx * k_h_stride + cu_k_start * k_s_stride
        k = tl.load(
            k_base + offs_n[:, None] * k_s_stride + offs_d[None, :] * k_d_stride,
            mask=k_mask[:, None],
            other=0.0,
        )
        v_base = v_ptr + head_idx * v_h_stride + cu_k_start * v_s_stride
        v = tl.load(
            v_base + offs_n[:, None] * v_s_stride + offs_d[None, :] * v_d_stride,
            mask=k_mask[:, None],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_base = o_ptr + head_idx * o_h_stride + cu_q_start * o_s_stride
    tl.store(
        o_base + offs_m[:, None] * o_s_stride + offs_d[None, :] * o_d_stride,
        acc.to(tl.bfloat16),
        mask=offs_m[:, None] < seq_q_len,
    )


def flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float = None,
) -> torch.Tensor:
    """
    Varlen FlashAttention for privateuse1 (燧原)
    Args:
        q: [total_q, n_heads, head_dim] BF16
        k: [total_k, n_heads, head_dim] BF16
        v: [total_k, n_heads, head_dim] BF16
        cu_seqlens_q: [batch+1], int32, cumulative sequence length of q
        cu_seqlens_k: [batch+1], int32, cumulative sequence length of k
        max_seqlen_q: max sequence length in q
        max_seqlen_k: max sequence length in k
    """
    assert q.device.type == "privateuse1"
    assert q.dtype == torch.bfloat16
    n_heads = q.shape[1]
    head_dim = q.shape[2]
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    out = torch.empty_like(q)
    batch = cu_seqlens_q.shape[0] - 1

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = head_dim

    grid = (batch, n_heads)
    _flash_attn_varlen_kernel[grid](
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q.stride(1),
        q.stride(0),
        q.stride(2),
        k.stride(1),
        k.stride(0),
        k.stride(2),
        v.stride(1),
        v.stride(0),
        v.stride(2),
        out.stride(1),
        out.stride(0),
        out.stride(2),
        softmax_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out
