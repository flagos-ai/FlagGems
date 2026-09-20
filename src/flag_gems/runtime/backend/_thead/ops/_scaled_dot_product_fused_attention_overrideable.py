import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# General (any-layout) fused SDPA forward kernel.
# Semantics: out = (softmax(QK^T * scale + attn_bias (+ causal mask))) @ V
# Matches FlagGems fused attention: fp32 accumulation, p cast to input dtype
# for the PV dot, and a base-2 logsumexp output (m + log2(l)).
# ---------------------------------------------------------------------------
@triton.jit
def _sdpa_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    o_ptr,
    lse_ptr,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    sbb,
    sbh,
    sbl,
    sbs,
    sob,
    soh,
    som,
    sod,
    H,
    L,
    S,
    D,
    DV,
    scale,
    dropout_p,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DO_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    m_valid = offs_m < L
    d_valid = offs_d < D
    dv_valid = offs_dv < DV

    q_base = q_ptr + b * sqb + h * sqh
    k_base = k_ptr + b * skb + h * skh
    v_base = v_ptr + b * svb + h * svh

    q_ptrs = q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    q_in = tl.load(q_ptrs, mask=m_valid[:, None] & d_valid[None, :], other=0.0)
    q = q_in

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < S

        k_ptrs = k_base + offs_n[:, None] * skn + offs_d[None, :] * skd
        k_in = tl.load(k_ptrs, mask=n_valid[:, None] & d_valid[None, :], other=0.0)
        k_t = k_in

        qk = tl.dot(q, tl.trans(k_t), allow_tf32=False)
        qk = qk * scale * 1.4426950408889634  # fold LOG2E for exp2 softmax

        if HAS_BIAS:
            bias_ptrs = (
                b_ptr
                + b * sbb
                + h * sbh
                + offs_m[:, None] * sbl
                + offs_n[None, :] * sbs
            )
            bias = tl.load(
                bias_ptrs, mask=m_valid[:, None] & n_valid[None, :], other=0.0
            )
            qk = qk + bias

        if IS_CAUSAL:
            valid = (offs_m[:, None] >= offs_n[None, :]) & n_valid[None, :]
        else:
            valid = n_valid[None, :]
        qk = tl.where(valid, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])

        if DO_DROPOUT:
            r = tl.rand(0, offs_m[:, None] * S + offs_n[None, :])
            keep = (r > dropout_p).to(tl.float32) / (1.0 - dropout_p)
            p = p * keep

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = v_base + offs_n[:, None] * svn + offs_dv[None, :] * svd
        v_in = tl.load(v_ptrs, mask=n_valid[:, None] & dv_valid[None, :], other=0.0)
        v_t = v_in
        p_d = p.to(v_in.dtype)

        acc = acc * alpha[:, None] + tl.dot(p_d, v_t, allow_tf32=False)
        m_i = m_ij

    acc = acc / l_i[:, None]

    out_ptrs = (
        o_ptr + b * sob + h * soh + offs_m[:, None] * som + offs_dv[None, :] * sod
    )
    tl.store(out_ptrs, acc.to(q_in.dtype), mask=m_valid[:, None] & dv_valid[None, :])

    # base-2 logsumexp (FlagGems convention: m + log2(l))
    lse = m_i + tl.log2(l_i)
    lse_ptrs = lse_ptr + (b * H + h) * L + offs_m
    tl.store(lse_ptrs, lse, mask=m_valid)


# ---------------------------------------------------------------------------
# Contiguous fast path: q,k,v,o are contiguous (B,H,L,D)/(B,H,S,D)/(B,H,S,DV).
# H, L, S, D, DV are compile-time so masks fold away when shapes are block
# multiples and launch marshaling drops to 9 runtime args.
# ---------------------------------------------------------------------------
@triton.jit
def _sdpa_fwd_c(
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    o_ptr,
    lse_ptr,
    sb0,
    sb1,
    sb2,
    sb3,
    H: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    scale,
    dropout_p,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DO_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)

    m_valid = offs_m < L
    d_valid = offs_d < D
    dv_valid = offs_dv < DV

    bh = b * H + h
    q_base = q_ptr + bh * L * D
    k_base = k_ptr + bh * S * D
    v_base = v_ptr + bh * S * DV
    o_base = o_ptr + bh * L * DV

    q_ptrs = q_base + offs_m[:, None] * D + offs_d[None, :]
    q_in = tl.load(q_ptrs, mask=m_valid[:, None] & d_valid[None, :], other=0.0)
    q = q_in

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < S

        k_ptrs = k_base + offs_n[:, None] * D + offs_d[None, :]
        k_in = tl.load(k_ptrs, mask=n_valid[:, None] & d_valid[None, :], other=0.0)
        k_t = k_in

        qk = tl.dot(q, tl.trans(k_t), allow_tf32=False)
        qk = qk * scale * 1.4426950408889634  # fold LOG2E for exp2 softmax

        if HAS_BIAS:
            bias_ptrs = (
                b_ptr
                + b * sb0
                + h * sb1
                + offs_m[:, None] * sb2
                + offs_n[None, :] * sb3
            )
            bias = tl.load(
                bias_ptrs, mask=m_valid[:, None] & n_valid[None, :], other=0.0
            )
            qk = qk + bias

        if IS_CAUSAL:
            valid = (offs_m[:, None] >= offs_n[None, :]) & n_valid[None, :]
        else:
            valid = n_valid[None, :]
        qk = tl.where(valid, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])

        if DO_DROPOUT:
            r = tl.rand(0, offs_m[:, None] * S + offs_n[None, :])
            keep = (r > dropout_p).to(tl.float32) / (1.0 - dropout_p)
            p = p * keep

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = v_base + offs_n[:, None] * DV + offs_dv[None, :]
        v_in = tl.load(v_ptrs, mask=n_valid[:, None] & dv_valid[None, :], other=0.0)
        v_t = v_in
        p_d = p.to(v_in.dtype)

        acc = acc * alpha[:, None] + tl.dot(p_d, v_t, allow_tf32=False)
        m_i = m_ij

    acc = acc / l_i[:, None]

    out_ptrs = o_base + offs_m[:, None] * DV + offs_dv[None, :]
    tl.store(out_ptrs, acc.to(q_in.dtype), mask=m_valid[:, None] & dv_valid[None, :])

    # base-2 logsumexp (FlagGems convention: m + log2(l))
    lse = m_i + tl.log2(l_i)
    lse_ptrs = lse_ptr + bh * L + offs_m
    tl.store(lse_ptrs, lse, mask=m_valid)


def _bias_strides(attn_bias):
    nd = attn_bias.dim()
    return (0,) * (4 - nd) + tuple(attn_bias.stride())


# Constant return-tensor caches (values depend only on shape/device/dtype, not
# on input data). Built once per key to avoid per-call host->device transfers.
_CUM_Q_CACHE = {}
_CUM_K_CACHE = {}
_PHILOX_CACHE = {}
_DEBUG_EMPTY_CACHE = {}
_DEBUG_ZEROS_CACHE = {}


def _cached_cum(b, seq, device, cache):
    key = (device, b, seq)
    t = cache.get(key)
    if t is None:
        t = torch.tensor([0, seq] * b, dtype=torch.int32, device=device).reshape(b, 2)
        cache[key] = t
    return t


def _cached_philox(device):
    t = _PHILOX_CACHE.get(device)
    if t is None:
        t = torch.tensor([0], dtype=torch.int64, device=device)
        _PHILOX_CACHE[device] = t
    return t


def _cached_debug_empty(device, dtype):
    key = (device, dtype)
    t = _DEBUG_EMPTY_CACHE.get(key)
    if t is None:
        t = torch.empty(0, device=device, dtype=dtype)
        _DEBUG_EMPTY_CACHE[key] = t
    return t


def _cached_debug_zeros(device, dtype, shape):
    key = (device, dtype, shape)
    t = _DEBUG_ZEROS_CACHE.get(key)
    if t is None:
        t = torch.zeros(shape, device=device, dtype=dtype)
        _DEBUG_ZEROS_CACHE[key] = t
    return t


def _scaled_dot_product_fused_attention_overrideable(
    query,
    key,
    value,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    scale=None,
):
    q = query
    k = key
    v = value

    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    B, H, L, D = q.shape
    S = k.shape[-2]
    DV = v.shape[-1]

    if scale is None:
        scale = D**-0.5
    do_dropout = 0.0 < dropout_p < 1.0

    out = torch.empty((B, H, L, DV), device=q.device, dtype=q.dtype)
    lse = torch.empty((B, H, L), device=q.device, dtype=torch.float32)

    # Shape/dtype-adaptive tiling (CUDA-graph GPU sweep refined against eval).
    # 16-bit: L<=64 -> (64,64,w8,s3); L==128 -> (64,64,w4|w8,s2|s3) by S
    # (S==64: w4,s2; S==128: w8,s3); L>=256 -> (64,64,w4,s3)
    # fp32:   L<=64 -> (64,64,w8,s3); L==128 -> (64,128|64,w8,s3); L>=256 -> (64,128,w8,s3)
    is_16b = q.dtype in (torch.float16, torch.bfloat16)
    if L <= 64:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 8, 3
    elif L <= 128:
        if is_16b:
            BLOCK_M, BLOCK_N = 64, 64
            num_warps = 4 if S == 64 else 8
            num_stages = 2 if S == 64 else 3
        else:
            BLOCK_M, BLOCK_N, num_warps, num_stages = 64, (128 if S == 64 else 64), 8, 3
    else:
        BLOCK_M, BLOCK_N = 64, (64 if is_16b else 128)
        num_warps = 4 if is_16b else 8
        num_stages = 3
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_DV = triton.next_power_of_2(DV)

    grid = (triton.cdiv(L, BLOCK_M), B * H)

    if attn_bias is not None:
        st = _bias_strides(attn_bias)
        bias_ptr = attn_bias
    else:
        st = (0, 0, 0, 0)
        bias_ptr = q

    if (
        q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and out.is_contiguous()
    ):
        _sdpa_fwd_c[grid](
            q,
            k,
            v,
            bias_ptr,
            out,
            lse,
            st[0],
            st[1],
            st[2],
            st[3],
            H,
            L,
            S,
            D,
            DV,
            float(scale),
            float(dropout_p),
            HAS_BIAS=attn_bias is not None,
            IS_CAUSAL=is_causal,
            DO_DROPOUT=do_dropout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=BLOCK_DV,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _sdpa_fwd[grid](
            q,
            k,
            v,
            bias_ptr,
            out,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            st[0],
            st[1],
            st[2],
            st[3],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            H,
            L,
            S,
            D,
            DV,
            float(scale),
            float(dropout_p),
            HAS_BIAS=attn_bias is not None,
            IS_CAUSAL=is_causal,
            DO_DROPOUT=do_dropout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=BLOCK_DV,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if return_debug_mask:
        debug_mask = _cached_debug_zeros(q.device, q.dtype, (B, H, L, S))
    else:
        debug_mask = _cached_debug_empty(q.device, q.dtype)

    return (
        out,
        lse,
        _cached_cum(B, L, q.device, _CUM_Q_CACHE),
        _cached_cum(B, S, k.device, _CUM_K_CACHE),
        L,
        S,
        _cached_philox(q.device),
        _cached_philox(q.device),
        debug_mask,
    )
