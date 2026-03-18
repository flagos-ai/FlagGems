"""
Pure PyTorch reference for sparse MLA forward (vLLM 3D interface).

Used for correctness verification. Chunked to handle large configs without OOM.
"""

import torch


def ref_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v):
    """Pure PyTorch sparse MLA reference (vLLM-compatible 3D interface).

    Args:
        q: [s_q, h_q, d_qk], bfloat16/float
        kv: [s_kv, h_kv, d_qk], bfloat16/float
        indices: [s_q, h_kv, topk], int32
        sm_scale: float
        d_v: int

    Returns:
        (output, max_logits, lse)
        - output: [s_q, h_q, d_v], same dtype as q
        - max_logits: [s_q, h_q], float32 (zeros)
        - lse: [s_q, h_q], float32
    """
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, _ = kv.shape
    heads_per_group = h_q // h_kv

    q_f = q.float()
    kv_f = kv.float()

    output = torch.zeros(s_q, h_q, d_v, dtype=torch.float32, device=q.device)
    lse = torch.zeros(s_q, h_q, dtype=torch.float32, device=q.device)

    # Chunk over s_q to avoid OOM on large configs
    CHUNK = min(64, s_q)

    for g in range(h_kv):
        h_start = g * heads_per_group
        h_end = (g + 1) * heads_per_group

        for sq_start in range(0, s_q, CHUNK):
            sq_end = min(sq_start + CHUNK, s_q)

            # Gather KV by indices: [chunk, topk, d_qk]
            idx = indices[sq_start:sq_end, g, :].long()
            kv_gathered = kv_f[idx, g, :]

            # Q for this group: [chunk, heads_per_group, d_qk]
            q_chunk = q_f[sq_start:sq_end, h_start:h_end, :]

            # Attention scores: [chunk, heads_per_group, topk]
            scores = torch.einsum("shd,std->sht", q_chunk, kv_gathered)
            scores = scores * sm_scale

            # Softmax
            max_s = scores.max(dim=-1, keepdim=True).values
            exp_s = torch.exp(scores - max_s)
            sum_exp = exp_s.sum(dim=-1, keepdim=True)
            attn = exp_s / sum_exp

            # Weighted sum of values: [chunk, heads_per_group, d_v]
            v_gathered = kv_gathered[:, :, :d_v]
            out_chunk = torch.einsum("sht,std->shd", attn, v_gathered)

            output[sq_start:sq_end, h_start:h_end, :] = out_chunk
            lse[sq_start:sq_end, h_start:h_end] = max_s.squeeze(-1) + torch.log(
                sum_exp.squeeze(-1)
            )

    max_logits = torch.zeros(s_q, h_q, dtype=torch.float32, device=q.device)
    return output.to(q.dtype), max_logits, lse
