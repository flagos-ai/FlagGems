import logging
import triton
import triton.language as tl
import torch
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["scale", "topk", "total_bh"])
def fused_attn_kernel(
    Q, GKV, Out, attn_sink,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_gkvbm, stride_gkvt, stride_gkvd,
    stride_ob, stride_om, stride_oh, stride_od,
    scale, topk, total_bh,
    D: tl.constexpr,
    H: tl.constexpr,
    GRID_BH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_d = tl.arange(0, D)

    for bh_idx in tl.range(pid_bh, total_bh, GRID_BH):
        pid_b = bh_idx // H
        pid_h = bh_idx - pid_b * H

        q_base = Q + pid_b * stride_qb + pid_m * stride_qm + pid_h * stride_qh
        q_vec = tl.load(q_base + offs_d * stride_qd)
        q_scaled_bf = (q_vec.to(tl.float32) * scale).to(tl.bfloat16)

        gkv_offset = (pid_b + pid_m) * stride_gkvbm

        acc_o = tl.zeros([D], dtype=tl.float32)
        score_max = tl.full([], float("-inf"), dtype=tl.float32)
        sum_exp = tl.zeros([], dtype=tl.float32)

        offs_blk = tl.arange(0, BLOCK)
        num_blocks = (topk + BLOCK - 1) // BLOCK

        for t in range(num_blocks):
            block_start = t * BLOCK
            valid_mask = (block_start + offs_blk) < topk

            kv_ptrs = GKV + gkv_offset + (block_start + offs_blk[:, None]) * stride_gkvt + offs_d[None, :] * stride_gkvd
            kv_block = tl.load(kv_ptrs, mask=valid_mask[:, None], other=0.0)

            scores = tl.sum(q_scaled_bf[None, :] * kv_block, axis=1).to(tl.float32)
            scores = tl.where(valid_mask, scores, float("-inf"))

            block_max = tl.max(scores)
            score_max_prev = score_max
            score_max = tl.maximum(score_max, block_max)

            correction = tl.exp(score_max_prev - score_max)
            p = tl.exp(scores - score_max)

            acc_o = acc_o * correction
            p_bf = p.to(tl.bfloat16)
            acc_o += tl.sum(p_bf[:, None] * kv_block, axis=0).to(tl.float32)
            sum_exp = sum_exp * correction + tl.sum(p)

        sink_val = tl.load(attn_sink + pid_h)
        sum_exp = sum_exp + tl.exp(sink_val - score_max)
        acc_o = acc_o / sum_exp

        o_base = Out + pid_b * stride_ob + pid_m * stride_om + pid_h * stride_oh
        tl.store(o_base + offs_d * stride_od, acc_o.to(tl.bfloat16))


def sparse_attn_triton(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    o = torch.empty_like(q)

    if m == 1:
        idx = topk_idxs.view(b, topk, 1).expand(b, topk, d).long()
        gathered_kv = torch.gather(kv, 1, idx)
    else:
        idx_exp = topk_idxs.unsqueeze(-1).expand(b, m, topk, d).long()
        kv_exp = kv.unsqueeze(1).expand(b, m, -1, d)
        gathered_kv = torch.gather(kv_exp, 2, idx_exp)
    gathered_kv = gathered_kv.reshape(b * m, topk, d)

    BLOCK = 128
    total_bh = b * h
    GRID_BH = min(total_bh, 255)

    grid = (m, GRID_BH)

    with torch_device_fn.device(q.device):
        fused_attn_kernel[grid](
            q, gathered_kv, o, attn_sink,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            gathered_kv.stride(0), gathered_kv.stride(1), gathered_kv.stride(2),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            softmax_scale, topk, total_bh,
            D=d, H=h, GRID_BH=GRID_BH, BLOCK=BLOCK,
            num_warps=1,
        )
    return o
