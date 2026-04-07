import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: sparse attention with attention-sink
# grid = (m, b)  — one program per (seq_pos, batch), handles ALL heads
# Aligned with tilelang version: uses tl.dot (GEMM) instead of vector dot
#
# Iluvatar-compatible: no tl.load mask/other, no tl.where
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
    H_ACTUAL,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # ---- load Q matrix: (H, D) — all heads at once ----
    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    h_mask_f = h_mask.to(tl.float32)
    # Use offs_h directly, will load OOB for h >= H_ACTUAL but we mask later
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs)  # (H, D) bf16
    # zero padded heads via arithmetic (avoid tl.where)
    q_block = (q_block.to(tl.float32) * h_mask_f[:, None]).to(tl.bfloat16)

    # ---- base pointers ----
    kv_base = KV + pid_b * stride_kvb
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    # ---- online softmax state ----
    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (topk + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        # -- gather indices (clamp to avoid OOB, mask via score bias) --
        raw_offs = t * BLOCK + offs_blk               # (BLOCK,)
        idx_mask = raw_offs < topk
        safe_raw_offs = tl.minimum(raw_offs, topk - 1)
        idxs = tl.load(idx_base + safe_raw_offs * stride_idxk)  # (BLOCK,)

        # -- gather KV block: (BLOCK, D) --
        safe_idxs = tl.maximum(idxs, 0)
        kv_ptrs = kv_base + safe_idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv_block = tl.load(kv_ptrs)  # (BLOCK, D) bf16

        # -- scores: Q @ KV^T -> (H, BLOCK) via GEMM --
        acc_s = tl.dot(q_block, tl.trans(kv_block))   # (H, D) @ (D, BLOCK) = (H, BLOCK)
        acc_s = acc_s * scale
        # mask invalid positions to -large via arithmetic (avoid tl.where)
        mask_bias = (idx_mask.to(tl.float32) - 1.0) * 1e30  # 0 for valid, -1e30 for invalid
        acc_s = acc_s + mask_bias[None, :]             # broadcast: (H, BLOCK)

        # -- online softmax update --
        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)              # (H,)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)  # (H,)
        p = tl.exp(acc_s - scores_max[:, None])        # (H, BLOCK)

        # -- accumulate output: acc_o = acc_o * correction + P @ KV --
        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_block)  # (H, BLOCK) @ (BLOCK, D) = (H, D)

        scores_sum = tl.sum(p, axis=1)                 # (H,)
        sum_exp = sum_exp * correction + scores_sum

    # ---- incorporate attn_sink ----
    # attn_sink is now padded to H elements, safe to load with offs_h
    sink_vals = tl.load(attn_sink + offs_h)  # (H,)
    # zero padded heads' sink via arithmetic
    sink_vals = sink_vals * h_mask_f
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    # ---- normalize ----
    acc_o = acc_o / sum_exp[:, None]

    # ---- store output: (H, D) ----
    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


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

    # H must be >= 16 for tl.dot; pad to next power of 2
    H_padded = max(16, triton.next_power_of_2(h))

    # Pad attn_sink to H_padded elements for safe kernel indexing
    if attn_sink.shape[0] < H_padded:
        attn_sink_padded = torch.zeros(H_padded, dtype=attn_sink.dtype, device=attn_sink.device)
        attn_sink_padded[:attn_sink.shape[0]] = attn_sink
    else:
        attn_sink_padded = attn_sink

    # Reduce BLOCK for large D to stay within resource limits
    if d >= 256:
        BLOCK = 16
    else:
        BLOCK = 64

    # Reduce warps for large D to lower register pressure
    num_warps = 2 if d >= 256 else 8

    grid = (m, b)  # each program handles ALL h heads
    sparse_attn_triton_kernel[grid](
        q, kv, o, attn_sink_padded, topk_idxs,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        topk_idxs.stride(0), topk_idxs.stride(1), topk_idxs.stride(2),
        softmax_scale,
        topk,
        h,
        BLOCK=BLOCK,
        D=d,
        H=H_padded,
        num_warps=num_warps,
        num_stages=1,
    )
    return o
