import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: sparse attention with attention-sink
# grid = (m, b, h)  — one program per (seq_pos, batch, head)
# 昆仑芯适配版本
# ---------------------------------------------------------------------------
@triton.jit
def sparse_attn_triton_kernel(
    Q,          # (b, m, h, d)  bf16
    KV,         # (b, n, d)     bf16
    O,          # (b, m, h, d)  bf16
    attn_sink,  # (h,)          fp32
    topk_idxs,  # (b, m, topk)  int32
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kvb, stride_kvn, stride_kvd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_idxb, stride_idxm, stride_idxk,
    scale,
    topk,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # ---- load Q vector: (D,) for this head ----
    q_base = Q + pid_b * stride_qb + pid_m * stride_qm + pid_h * stride_qh
    offs_d = tl.arange(0, D)
    q_vec = tl.load(q_base + offs_d * stride_qd)  # (D,) bf16

    # ---- base pointers ----
    kv_base = KV + pid_b * stride_kvb
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    # ---- online softmax state ----
    acc_o = tl.zeros([D], dtype=tl.float32)
    score_max = float("-inf")
    sum_exp = 0.0

    # Process each topk element one by one
    for k in range(topk):
        # -- gather KV vector --
        idx = tl.load(idx_base + k * stride_idxk)     # scalar
        # Handle negative indices (padding values like -1): clamp to 0
        idx = tl.where(idx < 0, 0, idx)

        # Load KV for this index: (D,)
        kv_ptrs = kv_base + idx * stride_kvn + offs_d * stride_kvd
        kv_vec = tl.load(kv_ptrs)  # (D,)

        # -- compute score using element-wise multiply then sum --
        # This is equivalent to dot product for 1D vectors
        score = tl.sum(q_vec * kv_vec)

        score = score * scale

        # -- online softmax update --
        score_max_prev = score_max
        score_max = tl.maximum(score_max, score)

        correction = tl.exp(score_max_prev - score_max)
        p = tl.exp(score - score_max)

        # -- accumulate output: acc_o = acc_o * correction + p * kv_vec --
        acc_o = acc_o * correction + p * kv_vec.to(tl.float32)

        sum_exp = sum_exp * correction + p

    # ---- incorporate attn_sink ----
    sink_val = tl.load(attn_sink + pid_h)              # scalar
    sum_exp = sum_exp + tl.exp(sink_val - score_max)

    # ---- normalize ----
    acc_o = acc_o / sum_exp

    # ---- store output: (D,) ----
    o_base = O + pid_b * stride_ob + pid_m * stride_om + pid_h * stride_oh
    o_ptrs = o_base + offs_d * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    o = torch.empty_like(q)

    grid = (m, b, h)  # each program handles one (seq_pos, batch, head)
    sparse_attn_triton_kernel[grid](
        q, kv, o, attn_sink, topk_idxs,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        topk_idxs.stride(0), topk_idxs.stride(1), topk_idxs.stride(2),
        softmax_scale,
        topk,
        D=d,
        num_warps=2,
    )
    return o