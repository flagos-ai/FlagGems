import torch
import triton
import triton.language as tl

spar_mla_fwd_configs = [
    triton.Config({"num_stages": 4, "num_warps": 8}),
    triton.Config({"num_stages": 2, "num_warps": 4}),
]


@triton.autotune(  # Decorate the kernel
    configs=spar_mla_fwd_configs,
    key=["K", "is_causal"],
)
@triton.jit
def triton_sparse_mla_fwd(
    q,
    kv,
    indices,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kvb,
    stride_kvg,
    stride_kvn,
    stride_kvd,
    stride_tb,
    stride_tg,
    stride_tm,
    stride_tt,  # indices dim
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    SQ: tl.constexpr,  # seqlen
    K: tl.constexpr,  # topk
    D: tl.constexpr,  # QKV dim
    TD: tl.constexpr,  # tail dim
    DP: tl.constexpr,
    TDP: tl.constexpr,
    G: tl.constexpr,  # group_size
    BK: tl.constexpr,
    BH: tl.constexpr,
    is_causal: tl.constexpr,
):
    i_b, i_sq, i_gbh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_g, i_bh = i_gbh // G, i_gbh % G
    q_base = q + i_b * stride_qb + i_sq * stride_qm + i_gbh * (BH * stride_qh)
    tq_base = q_base + D * stride_qd
    kv_base = kv + i_b * stride_kvb + i_g * stride_kvg
    tkv_base = kv_base + D * stride_kvd
    t_base = indices + i_b * stride_tb + i_sq * stride_tm + i_g * stride_tg
    o_base = output + i_b * stride_ob + i_sq * stride_om + i_gbh * (BH * stride_oh)
    l_base = lse + i_b * stride_lb + i_sq * stride_lm + i_gbh * (BH * stride_lh)

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, DP)
    offs_td = tl.arange(0, TDP)
    offs_od = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_h = i_bh * BH + offs_h < G
    mask_d = offs_d < D
    mask_td = offs_td < TD
    mask_od = mask_d

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_msk = mask_h[:, None] & mask_d[None, :]
    q_blk = tl.load(q_ptr, q_msk, other=0.0).to(tl.float16)

    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :] * stride_qd
    tq_msk = mask_h[:, None] & mask_td[None, :]
    tq_blk = tl.load(tq_ptr, tq_msk, other=0.0).to(tl.float16)

    max_log = tl.full([BH], float("-inf"), dtype=tl.float16)
    sum_exp = tl.full([BH], 1.0, dtype=tl.float16)
    acc = tl.zeros([BH, DP], dtype=tl.float16)
    qk = tl.zeros([BH, BK], dtype=tl.float16)

    log_scale: tl.constexpr = sm_scale * 1.44269504

    # max_col = max(0, i_sq + SKV - SQ) if is_causal else SKV-1
    max_col = i_sq if is_causal else SQ - 1

    NK = tl.cdiv(K, BK)
    for ck in range(NK):
        t_ptr = (BK * ck + offs_t) * stride_tt
        t_msk = t_ptr < K
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids <= max_col) & (kv_ids >= 0)

        # if mask_ids.max(0) > 0:
        if ck * BK <= max_col:
            kv_ptr = (
                kv_base + offs_d[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
            )
            kv_msk = mask_d[:, None] & mask_ids[None, :]
            kv_blk = tl.load(kv_ptr, kv_msk, other=0.0).to(tl.float16)  # [DP, BK]

            tkv_ptr = (
                tkv_base + offs_td[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
            )
            tkv_msk = mask_td[:, None] & mask_ids[None, :]
            tkv_blk = tl.load(tkv_ptr, tkv_msk, other=0.0).to(tl.float16)  # [TDP, BK]

            qk = tl.dot(q_blk, kv_blk, out_dtype=tl.float16)
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float16) * log_scale
            # qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float16) * sm_scale

            qk = tl.where(mask_ids[None, :], qk, float("-inf"))  # [BH, BK]

            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp2(qk - new_max[:, None]).to(tl.float16)
            # exp_qk = tl.math.exp(qk - new_max[:, None]).to(tl.float16)
            sum_qk = tl.sum(exp_qk, axis=1)
            alpha = tl.math.exp2(max_log - new_max).to(tl.float16)
            # alpha = tl.math.exp(max_log - new_max).to(tl.float16)
            sum_exp = sum_exp * alpha + sum_qk
            acc = acc * alpha[:, None]
            acc = tl.dot(
                exp_qk, kv_blk.trans(), acc, out_dtype=tl.float16
            )  # [BH, BK] @ [BK, DP] = [BH, DP]

            max_log = new_max.to(tl.float16)

    out_vals = acc / sum_exp[:, None]
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_od[None, :] * stride_od
    o_msk = mask_h[:, None] & mask_od[None, :]
    # o_msk &= tl.zeros_like(o_msk)
    tl.store(o_ptr, out_vals.to(q_blk.dtype), o_msk)

    fin_log = max_log + tl.math.log2(sum_exp.to(tl.float32))  # return lse / ln2
    # fin_log *= 0.69314718
    # fin_log = max_log + tl.math.log(sum_exp.to(tl.float32))
    # fin_log *= 1.44269504 # return lse / ln2
    l_ptr = l_base + offs_h * stride_lh
    l_msk = mask_h
    tl.store(l_ptr, fin_log.to(q_blk.dtype), l_msk)


def triton_sparse_mla_fwd_interface(
    q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512
):
    is_causal = True
    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, _, VG, _ = kv.shape

    # assert DT == 576, "you should assign dim otherwise"
    D = d_v

    assert kv.shape[-1] == DT
    TD = DT - D
    DP = triton.next_power_of_2(D)
    TDP = triton.next_power_of_2(TD)
    assert kv.shape[0] == B
    _, _, _, K = indices.shape
    assert indices.shape == (B, SQ, VG, K)
    G = H // VG
    if sm_scale is None:
        sm_scale = DT**-0.5
    BH = max(16, min(64, triton.next_power_of_2(G)))
    NH = triton.cdiv(G, BH)
    BK = 32
    output = torch.zeros((B, SQ, H, D), device=q.device, dtype=q.dtype)
    lse = torch.full((B, SQ, H), float("-inf"), device=q.device, dtype=q.dtype)
    grid = (B, SQ, VG * NH)  # (SQ//BQ, B*H)
    triton_sparse_mla_fwd[grid](
        q,
        kv,
        indices,
        sm_scale,
        output,
        lse,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        q.stride(3),  # [B, H, SQ, DT]
        kv.stride(0),
        kv.stride(2),
        kv.stride(1),
        kv.stride(3),  # [B, VG, SKV, DT]
        indices.stride(0),
        indices.stride(2),
        indices.stride(1),
        indices.stride(3),  # [B, VG, SQ, K]
        output.stride(0),
        output.stride(2),
        output.stride(1),
        output.stride(3),  # [B, H, SQ, D]
        lse.stride(0),
        lse.stride(2),
        lse.stride(1),  # [B, H, SQ]
        SQ,
        K,
        D,
        TD,
        DP,
        TDP,
        G,
        BK,
        BH,
        is_causal,
    )
    return output, lse
