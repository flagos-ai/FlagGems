import os
os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'

import torch
import torch_npu
import triton
import triton.language as tl
import math


@triton.jit
def sparse_attn_triton_kernel(
    Q, KV, O, ATTN_SINK, TOPK_IDXS,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kvb, stride_kvn, stride_kvd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_ib, stride_im, stride_ik,
    scale,
    topk,
    num_blocks: tl.constexpr,
    BLOCK: tl.constexpr,
    H_TILE: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_ht = tl.program_id(2)

    h_start = pid_ht * H_TILE
    offs_h = h_start + tl.arange(0, H_TILE)
    offs_d = tl.arange(0, D)
    offs_block = tl.arange(0, BLOCK)

    q_ptrs = Q + pid_b * stride_qb + pid_m * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs).to(tl.bfloat16)

    acc_o = tl.zeros((H_TILE, D), dtype=tl.float32)
    sum_exp = tl.zeros((H_TILE,), dtype=tl.float32)
    scores_max = tl.full((H_TILE,), value=float('-inf'), dtype=tl.float32)

    for t in range(num_blocks):
        idx_offs = t * BLOCK + offs_block
        idx_ptrs = TOPK_IDXS + pid_b * stride_ib + pid_m * stride_im + idx_offs * stride_ik
        mask_valid_load = idx_offs < topk
        idxs = tl.load(idx_ptrs, mask=mask_valid_load, other=-1)

        valid = idxs != -1
        safe_idxs = tl.where(valid, idxs, 0)

        kv_ptrs = KV + pid_b * stride_kvb + safe_idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv = tl.load(kv_ptrs).to(tl.bfloat16)
        kv = tl.where(valid[:, None], kv, 0.0)

        acc_s = tl.dot(q, tl.trans(kv), out_dtype=tl.float32)
        acc_s = tl.where(valid[None, :], acc_s, float('-inf'))
        acc_s *= scale

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)
        scores_scale = tl.exp(scores_max_prev - scores_max)

        acc_s = tl.exp(acc_s - scores_max[:, None])
        scores_sum = tl.sum(acc_s, axis=1)
        sum_exp = sum_exp * scores_scale + scores_sum

        acc_s_cast = acc_s.to(tl.bfloat16)

        acc_o = acc_o * scores_scale[:, None]
        acc_o += tl.dot(acc_s_cast, kv, out_dtype=tl.float32)

    sink = tl.load(ATTN_SINK + offs_h)
    sum_exp += tl.exp(sink - scores_max)

    acc_o = acc_o / sum_exp[:, None]
    o = acc_o.to(tl.bfloat16)
    o_ptrs = O + pid_b * stride_ob + pid_m * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o)


def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, s, h, d = q.size()

    orig_h = h
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])
        h = 16

    topk = topk_idxs.shape[-1]
    block = 16
    num_blocks = math.ceil(topk / block)

    h_tile = 16
    n_h_tiles = h // h_tile

    o = torch.empty_like(q)

    grid = (s, b, n_h_tiles)
    sparse_attn_triton_kernel[grid](
        q, kv, o, attn_sink, topk_idxs,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        topk_idxs.stride(0), topk_idxs.stride(1), topk_idxs.stride(2),
        softmax_scale,
        topk,
        num_blocks=num_blocks,
        BLOCK=block,
        H_TILE=h_tile,
        D=d,
        num_warps=4,
        num_stages=1,
    )

    if orig_h < 16:
        o = o.narrow(2, 0, orig_h).contiguous()

    return o