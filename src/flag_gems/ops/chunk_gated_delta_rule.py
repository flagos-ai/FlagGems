import logging

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Triton kernel for one chunk of gated delta rule recurrence.
# Each program handles one (batch, head) pair.
# For each timestep t in [chunk_start, chunk_start + BT):
#   h = exp(g_t) * h + beta_t * k_t (x) (v_t - k_t^T @ h)
#   o_t = q_t^T @ h
@libentry()
@triton.jit
def chunk_gated_delta_rule_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    g_ptr,
    h_ptr,
    o_ptr,
    B,
    H,
    L,
    D_k,
    D_v,
    BT,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_hb,
    stride_hh,
    stride_hk,
    stride_hv,
    chunk_idx,
    BLOCK_DV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h_idx = pid_bh % H

    h_base = h_ptr + b * stride_hb + h_idx * stride_hh
    dv_offs = tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < D_v

    t_start = chunk_idx * BT
    for t_local in range(BT):
        t = t_start + t_local
        t_mask = t < L

        g_t = tl.load(g_ptr + b * H * L + h_idx * L + t, mask=t_mask, other=0.0).to(
            tl.float32
        )
        decay = tl.exp(tl.clamp(g_t, -10.0, 10.0))
        beta_t = tl.load(
            beta_ptr + b * H * L + h_idx * L + t, mask=t_mask, other=0.0
        ).to(tl.float32)

        q_base = q_ptr + b * stride_qb + h_idx * stride_qh + t * stride_ql
        k_base = k_ptr + b * stride_qb + h_idx * stride_qh + t * stride_ql
        v_base = v_ptr + b * stride_vb + h_idx * stride_vh + t * stride_vl

        # Load v_t [D_v]
        v_t = tl.load(v_base + dv_offs * stride_vd, mask=dv_mask, other=0.0).to(
            tl.float32
        )

        # proj = k_t^T @ h -> [D_v]
        proj = tl.zeros([BLOCK_DV], dtype=tl.float32)
        for dk in range(D_k):
            h_row = tl.load(
                h_base + dk * stride_hk + dv_offs * stride_hv,
                mask=dv_mask,
                other=0.0,
            ).to(tl.float32)
            k_dk = tl.load(k_base + dk * stride_qd).to(tl.float32)
            proj += k_dk * h_row

        v_new = v_t - proj

        # h = decay * h + beta * k_t (x) v_new
        for dk in range(D_k):
            h_row = tl.load(
                h_base + dk * stride_hk + dv_offs * stride_hv,
                mask=dv_mask,
                other=0.0,
            ).to(tl.float32)
            k_dk = tl.load(k_base + dk * stride_qd).to(tl.float32)
            h_new = decay * h_row + beta_t * k_dk * v_new
            tl.store(
                h_base + dk * stride_hk + dv_offs * stride_hv,
                h_new,
                mask=dv_mask & t_mask,
            )

        # o_t = q_t^T @ h -> [D_v]
        o_t = tl.zeros([BLOCK_DV], dtype=tl.float32)
        for dk in range(D_k):
            h_row = tl.load(
                h_base + dk * stride_hk + dv_offs * stride_hv,
                mask=dv_mask,
                other=0.0,
            ).to(tl.float32)
            q_dk = tl.load(q_base + dk * stride_qd).to(tl.float32)
            o_t += q_dk * h_row

        o_base = o_ptr + b * stride_vb + h_idx * stride_vh + t * stride_vl
        tl.store(o_base + dv_offs * stride_vd, o_t, mask=dv_mask & t_mask)


def chunk_gated_delta_rule(
    q,
    k,
    v,
    beta,
    g,
    BT=64,
    initial_state=None,
    output_final_state=False,
):
    logger.debug("GEMS CHUNK_GATED_DELTA_RULE")

    B, H, L, D_k = q.shape
    D_v = v.shape[-1]

    pad_len = (BT - L % BT) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))

    T = q.shape[2]
    num_chunks = T // BT
    scale = D_k**-0.5
    q = (q * scale).contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    g = g.contiguous()

    h = torch.zeros(B, H, D_k, D_v, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        h.copy_(initial_state.float())

    o = torch.empty_like(v)
    BLOCK_DV = triton.next_power_of_2(D_v)
    grid = (B * H,)

    with torch_device_fn.device(q.device):
        for c in range(num_chunks):
            chunk_gated_delta_rule_kernel[grid](
                q,
                k,
                v,
                beta,
                g,
                h,
                o,
                B,
                H,
                T,
                D_k,
                D_v,
                BT,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                h.stride(0),
                h.stride(1),
                h.stride(2),
                h.stride(3),
                c,
                BLOCK_DV=BLOCK_DV,
            )

    o = o[:, :, :L].to(q.dtype)
    final_state = h if output_final_state else None
    return o, final_state
