import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_scores_kernel(
    Q,
    KV,
    LOGITS,
    topk_idxs,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    scale,
    topk,
    TP,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
):
    i_bmh = tl.program_id(0)
    i_t = tl.program_id(1)
    i_h = i_bmh % H
    i_bm = i_bmh // H
    i_m = i_bm % M
    i_b = i_bm // M

    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + i_b * stride_qb + i_m * stride_qm + i_h * stride_qh + offs_d * stride_qd
    ).to(tl.float32)

    offs_k = i_t * BLOCK_K + tl.arange(0, BLOCK_K)
    ks = tl.minimum(offs_k, topk - 1)
    ids = tl.load(topk_idxs + i_b * stride_idxb + i_m * stride_idxm + ks * stride_idxk)
    valid = (offs_k < topk) & (ids >= 0)
    ids = tl.where(valid, ids, 0)

    kv = tl.load(
        KV
        + i_b * stride_kvb
        + ids[:, None] * stride_kvn
        + offs_d[None, :] * stride_kvd,
        mask=valid[:, None],
        other=0.0,
    )

    sc = tl.sum(q[None, :] * kv.to(tl.float32), axis=1)
    sc = sc * scale
    sc = tl.where(valid, sc, float("-inf"))

    tl.store(LOGITS + i_bmh * TP + offs_k, sc)


@triton.jit
def _sparse_attn_softmax_kernel(
    LOGITS,
    PROBS,
    attn_sink,
    TP,
    topk,
    H_ACTUAL,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    i_bmh = tl.program_id(0)
    i_h = i_bmh % H
    offs_k = tl.arange(0, BLOCK_K)
    lg_base = LOGITS + i_bmh * TP
    p_base = PROBS + i_bmh * TP
    n_blocks = (topk + BLOCK_K - 1) // BLOCK_K

    run_max = float("-inf")
    for t in range(n_blocks):
        k_offs = t * BLOCK_K + offs_k
        x = tl.load(lg_base + k_offs)
        x = tl.where(k_offs < topk, x, float("-inf"))
        run_max = tl.maximum(run_max, tl.max(x))

    if i_h < H_ACTUAL:
        sink_val = tl.load(attn_sink + i_h)
        run_sum = tl.exp(sink_val - run_max)
    else:
        run_sum = 0.0
    total_sum = run_sum
    for t in range(n_blocks):
        k_offs = t * BLOCK_K + offs_k
        x = tl.load(lg_base + k_offs)
        x = tl.where(k_offs < topk, x, float("-inf"))
        total_sum += tl.sum(tl.exp(x - run_max))

    lse = run_max + tl.math.log(total_sum)

    for t in range(n_blocks):
        k_offs = t * BLOCK_K + offs_k
        x = tl.load(lg_base + k_offs)
        x = tl.where(k_offs < topk, x, float("-inf"))
        p = tl.exp(x - lse)
        p = tl.where(k_offs < topk, p, 0.0)
        tl.store(p_base + k_offs, p.to(tl.bfloat16))


@triton.jit
def _sparse_attn_gather_td_kernel(
    KV,
    GKV_TD,
    topk_idxs,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    topk,
    TP,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    M: tl.constexpr,
):
    i_bm = tl.program_id(0).to(tl.int64)
    i_z = tl.program_id(1)
    i_d = i_z % (D // BD)
    i_t = i_z // (D // BD)
    i_b = i_bm // M
    i_m = i_bm % M

    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    in_range = offs_t < topk
    t_off = tl.minimum(offs_t, topk - 1)
    ids = tl.load(
        topk_idxs + i_b * stride_idxb + i_m * stride_idxm + t_off * stride_idxk
    )
    ids_safe = tl.where(in_range & (ids >= 0), ids, 0)

    v = tl.load(
        KV
        + i_b * stride_kvb
        + ids_safe[:, None] * stride_kvn
        + offs_d[None, :] * stride_kvd
    )

    tl.store(GKV_TD + i_bm * (TP * D) + offs_t[:, None] * D + offs_d[None, :], v)


@triton.jit
def _sparse_attn_pv_kernel(
    PROBS,
    GKV_TD,
    O,
    TP: tl.constexpr,
    D: tl.constexpr,
    BH: tl.constexpr,
    BT: tl.constexpr,
    BDV: tl.constexpr,
    AH: tl.constexpr,
):
    i_bm = tl.program_id(0).to(tl.int64)
    i_z = tl.program_id(1)
    i_v = i_z % (D // BDV)
    i_bh = i_z // (D // BDV)

    offs_h = i_bh * BH + tl.arange(0, BH)
    offs_t = tl.arange(0, BT)
    offs_v = i_v * BDV + tl.arange(0, BDV)

    p_base = PROBS + (i_bm * AH + offs_h[:, None]) * TP
    v_base = GKV_TD + i_bm * (TP * D)

    acc = tl.zeros([BH, BDV], dtype=tl.float32)
    for it in range(TP // BT):
        pb = tl.load(p_base + it * BT + offs_t[None, :])
        vb = tl.load(v_base + (it * BT + offs_t)[:, None] * D + offs_v[None, :])
        acc = tl.dot(pb, vb, acc, out_dtype=tl.float32)

    tl.store(
        O + (i_bm * AH + offs_h[:, None]) * D + offs_v[None, :], acc.to(tl.bfloat16)
    )


def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]

    BT = 64
    BD = 64
    BDV = 256
    BH = max(16, min(64, triton.next_power_of_2(h)))

    n_blocks = (topk + BT - 1) // BT
    tp = n_blocks * BT
    AH = ((h + BH - 1) // BH) * BH
    ND = (d + BD - 1) // BD
    NDV = (d + BDV - 1) // BDV
    bm = b * m

    logits = torch.zeros((bm * AH, tp), device=q.device, dtype=torch.float32)
    probs = torch.empty((bm * AH, tp), device=q.device, dtype=torch.bfloat16)
    gkv_td = torch.empty((bm, tp, d), device=q.device, dtype=torch.bfloat16)
    o_pad = torch.empty((bm * AH, d), device=q.device, dtype=torch.bfloat16)

    _sparse_attn_scores_kernel[(b * m * h, n_blocks)](
        q,
        kv,
        logits,
        topk_idxs,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        softmax_scale,
        topk,
        tp,
        D=d,
        BLOCK_K=BT,
        M=m,
        H=h,
        num_warps=8,
    )

    _sparse_attn_softmax_kernel[(bm * AH,)](
        logits,
        probs,
        attn_sink,
        tp,
        topk,
        h,
        H=AH,
        BLOCK_K=BT,
        num_warps=4,
    )

    _sparse_attn_gather_td_kernel[(bm, n_blocks * ND)](
        kv,
        gkv_td,
        topk_idxs,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        topk,
        tp,
        D=d,
        BT=BT,
        BD=BD,
        M=m,
    )

    _sparse_attn_pv_kernel[(bm, (AH // BH) * NDV)](
        probs,
        gkv_td,
        o_pad,
        TP=tp,
        D=d,
        BH=BH,
        BT=BT,
        BDV=BDV,
        AH=AH,
    )

    return o_pad.view(b, m, AH, d)[:, :, :h, :].contiguous()
