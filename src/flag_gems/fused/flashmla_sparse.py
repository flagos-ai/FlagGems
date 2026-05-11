from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=2),
        triton.Config({"BK": 64, "BH": 64}, num_warps=8, num_stages=4),
    ],
    key=["SQ", "HQ", "DQK", "SKV", "TOPK", "HAVE_ATTN_SINK", "HAVE_TOPK_LENGTH"],
)
@triton.jit
def triton_flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,  # s_q
    HQ: tl.constexpr,  # h_q=64 or 128
    DQK: tl.constexpr,  # d_qk=512 or 576
    SKV,  # s_kv
    TOPK: tl.constexpr,  # topk
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)  # prevent mul overflow
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)

    # `[BH, 256] x 2` delivers better performance than `[BH, 512]` when BH=64
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + offs_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        # step1: load indices
        t_ptr = BK * ck + offs_t  # [BK]
        t_msk = t_ptr < topk_len
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        # filter invalid index that may cause overflow in mul
        kv_ids = tl.where(mask_ids, kv_ids, 0)

        # step2: gather kv with indices
        kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")  # [BDP, BK]
        kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")  # [BDP, BK]
        # step3: (q @ kv) * sm_scale
        qk = tl.dot(
            q_blk0, kv_blk0, out_dtype=tl.float32
        )  # [BH, BDP]@[BDP, BK] -> [BH, BK]
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            tkv_ptr = tkv_base + offs_td[:, None] + kv_ids[None, :] * stride_kvn
            tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")  # [TDP, BK]
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale

        # step4: preprocess for logsumexp
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))  # [BH, BK]
        # step5: lse=logsumexp(qk), loop part
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))  # [BH]
        exp_qk = tl.math.exp(qk - new_max[:, None])  # [BH, BK]
        sum_qk = tl.sum(exp_qk, axis=1)  # [BH]
        alpha = tl.math.exp(max_log - new_max)  # [BH]
        sum_exp = sum_exp * alpha + sum_qk  # [BH]
        # step6: exp(qk-lse) @ gathered_kv.trans(), loop part
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )  # [BH, BK]@[BK, BDP]->[BH, BDP]
        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )  # [BH, BK]@[BK, BDP]->[BH, BDP]
        max_log = new_max

    # step7: store max_logits
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    tl.store(max_log_base + offs_h, max_log)  # [BH], float32

    # step8: lse=logsumexp(qk) final part, store lse
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h, lse_out)  # [BH], float32

    # step9: exp(qk-lse) @ gathered_kv.trans(), final part
    if HAVE_ATTN_SINK:
        # step10: attn_sink
        sink = tl.load(attn_sink_ptr + offs_h)  # [BH]
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp

    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)
    # step11: store output
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]  # [BH, BDP]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by
            exp(lse) / (exp(lse) + exp(attn_sink)). +-inf in attn_sink will be handled normally (i.e., -inf has no
            effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32.
            If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]],
            ignoring later k/v tokens (even if provided in indices). In extremely rare cases (topk_length provided,
            there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token
            containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape

    assert d_v == 512, "Unsupported d_v"
    DV = d_v

    assert kv.shape[-1] == DQK
    _, _, TOPK = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,), "topk_length error shape"

    # check from FlashMLA
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)

    def grid(META):
        return (triton.cdiv(HQ, META["BH"]) * SQ,)

    triton_flash_mla_sparse_fwd[grid](
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale,
        output,
        max_logits,
        lse,
        q.stride(1),
        q.stride(0),
        kv.stride(1),
        kv.stride(0),
        indices.stride(1),
        indices.stride(0),
        output.stride(1),
        output.stride(0),
        max_logits.stride(0),
        lse.stride(0),
        SQ,
        HQ,
        DQK,
        SKV,
        TOPK,
        attn_sink is not None,
        topk_length is not None,
    )
    return output, max_logits, lse
