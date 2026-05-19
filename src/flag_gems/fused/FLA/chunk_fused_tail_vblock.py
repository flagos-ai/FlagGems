# V-blocked fused tail for the official K=V=BT=64 chunk_gated_delta_rule path.

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry


_FUSED_TAIL_BV = 16


def can_use_fused_tail_vblock(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    if cu_seqlens is not None or initial_state is None or not output_final_state:
        return False
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 4 or u.ndim != 4 or g.ndim != 3:
        return False

    B, T, Hg, K = q.shape
    H, V = u.shape[2], u.shape[3]
    if k.shape != (B, T, Hg, K):
        return False
    if w.shape != (B, T, H, K) or g.shape != (B, T, H):
        return False
    if initial_state.shape != (B, H, K, V):
        return False
    if chunk_size != 64 or T % 64 != 0 or (K, V) != (64, 64) or H % Hg != 0:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not all(x.dtype == q.dtype for x in (k, w, u, g, initial_state)):
        return False
    return all(x.is_contiguous() for x in (q, k, w, u, g, initial_state))


@libentry()
@triton.jit
def _chunk_gated_delta_rule_fused_tail_vblock_kernel(
    q,
    k,
    w,
    u,
    g,
    h0,
    o,
    ht,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    Hg: tl.constexpr,
    BT: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    i_hg = i_h // (H // Hg)

    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    v_mask = offs_v < V

    h0_base = ((i_b * H + i_h) * K) * V
    h_acc = tl.load(
        h0 + h0_base + offs_k[:, None] * V + offs_v[None, :],
        mask=v_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    for i_t in range(0, tl.cdiv(T, BT)):
        t = i_t * BT + offs_t

        q_block = tl.load(
            q + (((i_b * T + t[:, None]) * Hg + i_hg) * K + offs_k[None, :])
        )
        k_t_block = tl.load(
            k + (((i_b * T + t[None, :]) * Hg + i_hg) * K + offs_k[:, None])
        )
        w_block = tl.load(
            w + (((i_b * T + t[:, None]) * H + i_h) * K + offs_k[None, :])
        )
        u_block = tl.load(
            u + (((i_b * T + t[:, None]) * H + i_h) * V + offs_v[None, :]),
            mask=v_mask[None, :],
            other=0.0,
        )
        g_vec = tl.load(g + (i_b * T + t) * H + i_h).to(tl.float32)

        residual = u_block.to(tl.float32) - tl.dot(w_block, h_acc.to(w_block.dtype))

        q_h = tl.dot(q_block, h_acc.to(q_block.dtype))
        qk = tl.dot(q_block, k_t_block).to(tl.float32)
        causal = offs_t[:, None] >= offs_t[None, :]
        qk = tl.where(causal, qk * tl.exp(g_vec[:, None] - g_vec[None, :]), 0.0)
        out = (
            q_h * tl.exp(g_vec)[:, None]
            + tl.dot(qk.to(u_block.dtype), residual.to(u_block.dtype))
        ) * scale
        tl.store(
            o + (((i_b * T + t[:, None]) * H + i_h) * V + offs_v[None, :]),
            out,
            mask=v_mask[None, :],
        )

        g_last = tl.load(g + (i_b * T + ((i_t + 1) * BT - 1)) * H + i_h).to(
            tl.float32
        )
        residual_for_state = residual * tl.exp(g_last - g_vec)[:, None]
        h_acc = h_acc * tl.exp(g_last) + tl.dot(
            k_t_block, residual_for_state.to(k_t_block.dtype)
        )

    ht_base = ((i_b * H + i_h) * K) * V
    tl.store(
        ht + ht_base + offs_k[:, None] * V + offs_v[None, :],
        h_acc,
        mask=v_mask[None, :],
    )


def chunk_gated_delta_rule_fused_tail_vblock(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K = q.shape
    H, V = u.shape[2], u.shape[3]

    o = torch.empty_like(u)
    final_state = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)
    _chunk_gated_delta_rule_fused_tail_vblock_kernel[
        (triton.cdiv(V, _FUSED_TAIL_BV), B * H)
    ](
        q,
        k,
        w,
        u,
        g,
        initial_state,
        o,
        final_state,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        BT=64,
        K=64,
        V=64,
        BV=_FUSED_TAIL_BV,
        num_warps=4,
        num_stages=3,
    )
    return o, final_state
