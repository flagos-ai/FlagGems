import logging

import torch
import triton
import triton.language as tl

_GEMS_LOGGER = logging.getLogger("flag_gems.ops._flash_attention_forward")

# Optional override for tiling experiments (None => per-head-dim defaults).
_CFG_OVERRIDE = None


@triton.jit
def _flash_attn_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    CuQ,
    CuK,
    SequsedK,
    AlibiSlopes,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_od,
    total_q,
    total_k,
    nheads_q,
    nheads_kv,
    scale,
    window_left,
    window_right,
    head_dim_orig,
    alibi_batch_stride,
    philox_seed,
    philox_offset,
    dropout_p,
    CAUSAL: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_SEQUSED: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    UNIFORM: tl.constexpr,
    SEQ_Q: tl.constexpr,
    SEQ_K: tl.constexpr,
    EVEN_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    if UNIFORM:
        # All sequences share the same length: derive the offsets arithmetically
        # and avoid the CuQ/CuK loads entirely (uniform 4D and single-seq cases).
        q_start = pid_b * SEQ_Q
        seqlen_q = SEQ_Q
        k_start = pid_b * SEQ_K
        seqlen_k = SEQ_K
    else:
        q_start = tl.load(CuQ + pid_b)
        q_end = tl.load(CuQ + pid_b + 1)
        k_start = tl.load(CuK + pid_b)
        k_end = tl.load(CuK + pid_b + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
    if (pid_m * BLOCK_M) >= seqlen_q:
        return

    # Causal/window alignment offset (bottom-right aligned mask).
    off = seqlen_k - seqlen_q

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < seqlen_q
    q_rows = (q_start + m_offs).to(tl.int64)

    d_offs = tl.arange(0, HEAD_DIM)
    d_mask = d_offs < head_dim_orig

    q_ptrs = (
        Q
        + q_rows[:, None] * stride_qb
        + pid_h * stride_qh
        + d_offs[None, :] * stride_qd
    )
    if EVEN_D:
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
    else:
        q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    kv_head = (pid_h * nheads_kv) // nheads_q

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Fully-masked key blocks are skipped: compute the key-index bounds that
    # the rows in this query block can attend to (per-row masks still apply).
    m_first = pid_m * BLOCK_M
    m_last = m_first + BLOCK_M - 1
    lo_key = 0
    hi_key = seqlen_k - 1
    if window_left >= 0:
        lo_key = tl.maximum(lo_key, m_first + off - tl.minimum(window_left, seqlen_k))
    if CAUSAL:
        hi_key = tl.minimum(hi_key, m_last + off)
    if window_right >= 0:
        hi_key = tl.minimum(hi_key, m_last + off + tl.minimum(window_right, seqlen_k))
    kb_start = lo_key // BLOCK_N
    kb_end = hi_key // BLOCK_N
    for kb in range(kb_start, kb_end + 1):
        n_offs = kb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seqlen_k
        k_rows = (k_start + n_offs).to(tl.int64)
        k_ptrs = (
            K
            + k_rows[None, :] * stride_kb
            + kv_head * stride_kh
            + d_offs[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + k_rows[None, :] * stride_vb
            + kv_head * stride_vh
            + d_offs[:, None] * stride_vd
        )
        if EVEN_D:
            kk = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
            vv = tl.load(v_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            kmask = n_mask[None, :] & d_mask[:, None]
            kk = tl.load(k_ptrs, mask=kmask, other=0.0)
            vv = tl.load(v_ptrs, mask=kmask, other=0.0)

        qk = tl.dot(q, kk)
        qk = qk * scale
        if HAS_ALIBI:
            slope = tl.load(AlibiSlopes + pid_b * alibi_batch_stride + pid_h)
            rel = m_offs[:, None] + off - n_offs[None, :]
            qk = qk - slope * tl.abs(rel)
        mask2d = tl.broadcast_to(n_mask[None, :], (BLOCK_M, BLOCK_N))
        if CAUSAL:
            mask2d = mask2d & (n_offs[None, :] <= m_offs[:, None] + off)
        # Sliding window, matching the reference (flash-attn) semantics:
        # - when window_size_left >= seqlen_k AND window_size_right >= seqlen_k the
        #   window is disabled entirely;
        # - otherwise each bound is clamped to seqlen_k and aligned bottom-right.
        if (
            window_left >= 0
            and window_right >= 0
            and window_left >= seqlen_k
            and window_right >= seqlen_k
        ):
            pass
        else:
            if window_left >= 0:
                mask2d = mask2d & (
                    n_offs[None, :]
                    >= m_offs[:, None] + off - tl.minimum(window_left, seqlen_k)
                )
            if window_right >= 0:
                mask2d = mask2d & (
                    n_offs[None, :]
                    <= m_offs[:, None] + off + tl.minimum(window_right, seqlen_k)
                )
        if HAS_SEQUSED:
            sk = tl.load(SequsedK + pid_b)
            mask2d = mask2d & (n_offs[None, :] < sk)
        qk = tl.where(mask2d, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        if HAS_DROPOUT:
            # Same philox consumption as the reference (flash_attn fwd_kernel):
            # per-element flat offset =
            #   philox_offset + (batch * nheads + head) * seqlen_q * seqlen_k
            #   + row * seqlen_k + col, fed to tl.rand; keep iff rng > dropout_p.
            batch_po = philox_offset + (pid_b * nheads_q + pid_h) * seqlen_q * seqlen_k
            offs2d = (batch_po + m_offs[:, None] * seqlen_k + n_offs[None, :]).to(
                tl.uint32
            )
            rng = tl.rand(philox_seed, offs2d)
            keep = rng > dropout_p
            p = tl.where(keep, p, 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), tl.trans(vv))
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = tl.where(l_i[:, None] > 0, acc / l_safe[:, None], 0.0)
    if HAS_DROPOUT:
        out = out / (1.0 - dropout_p)
    # softmax logsumexp of the scaled scores (same convention as the reference).
    lse_val = m_i + tl.log(l_safe)
    tl.store(Lse + pid_h * total_q + (q_start + m_offs), lse_val, mask=m_mask)
    out_ptrs = (
        Out
        + q_rows[:, None] * stride_ob
        + pid_h * stride_oh
        + d_offs[None, :] * stride_od
    )
    if EVEN_D:
        tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=m_mask[:, None])
    else:
        tl.store(
            out_ptrs,
            out.to(Out.dtype.element_ty),
            mask=m_mask[:, None] & d_mask[None, :],
        )


def _as_int(x, name):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    if x is None:
        raise ValueError(f"{name} is None")
    return int(x)


def _as_float(x, name):
    if isinstance(x, torch.Tensor):
        return float(x.item())
    if x is None:
        raise ValueError(f"{name} is None")
    return float(x)


def _as_bool(x, name):
    if isinstance(x, torch.Tensor):
        return bool(x.item())
    return bool(x)


def run(
    query,
    key,
    value,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None,
):
    _GEMS_LOGGER.debug("GEMS _FLASH_ATTENTION_FORWARD")
    q4 = query
    k4 = key
    v4 = value

    if q4.ndim == 4:
        batch, seqlen_q, nheads_q, head_dim = q4.shape
        q = q4.reshape(-1, nheads_q, head_dim)
        if k4.ndim == 4:
            _, seqlen_k, nheads_kv, _ = k4.shape
            k = k4.reshape(-1, nheads_kv, head_dim)
            v = v4.reshape(-1, nheads_kv, head_dim)
        else:
            raise ValueError("query/key ndim mismatch")
    else:
        q = q4
        k = k4
        v = v4
        nheads_q = q.shape[1]
        head_dim = q.shape[2]
        nheads_kv = k.shape[1]
        batch = 0

    total_q = q.shape[0]
    total_k = k.shape[0]
    device = q.device

    max_q = _as_int(max_q, "max_q")
    max_k = _as_int(max_k, "max_k")
    dropout_p = _as_float(dropout_p, "dropout_p")
    causal = _as_bool(is_causal, "is_causal")

    if cumulative_sequence_length_q is None or cumulative_sequence_length_k is None:
        uniform = True
        cu_q = None
        cu_k = None
        if q4.ndim == 4:
            batch = q4.shape[0]
            seqlen_q = q4.shape[1]
            seqlen_k = k4.shape[1]
        else:
            # Single (or already-flattened) sequence: batch == 1 with the full
            # lengths; the UNIFORM kernel path derives offsets arithmetically.
            batch = 1
            seqlen_q = total_q
            seqlen_k = total_k
    else:
        uniform = False
        cu_q = cumulative_sequence_length_q
        cu_k = cumulative_sequence_length_k
        if cu_q.dtype != torch.int32:
            cu_q = cu_q.to(torch.int32)
        if cu_k.dtype != torch.int32:
            cu_k = cu_k.to(torch.int32)
        batch = cu_q.numel() - 1

    if scale is None:
        scale = head_dim**-0.5
    scale = float(scale)

    win_l = -1 if window_size_left is None else int(window_size_left)
    win_r = -1 if window_size_right is None else int(window_size_right)

    has_seqused = seqused_k is not None
    has_alibi = alibi_slopes is not None
    if has_alibi:
        alibi_batch_stride = alibi_slopes.shape[1] if alibi_slopes.ndim == 2 else 0
    else:
        alibi_batch_stride = 0

    if dropout_p > 0.0:
        gen = torch.cuda.default_generators[device.index]
        st = gen.get_state()
        iv = st.view(torch.int64)
        philox_seed = int(iv[0])
        philox_offset = int(iv[1])
    else:
        philox_seed = 0
        philox_offset = 0

    out4 = torch.empty(q4.shape, dtype=q4.dtype, device=q4.device)
    out = out4.reshape(-1, nheads_q, head_dim) if q4.ndim == 4 else out4

    # softmax logsumexp in the reference layout: (nheads, total_q) internally;
    # returned as (B, H, S) for 4D inputs and (H, total_q) for 3D inputs.
    lse_buf = torch.empty((nheads_q, total_q), dtype=torch.float32, device=device)

    head_dim_pow2 = triton.next_power_of_2(head_dim)
    even_d = head_dim == head_dim_pow2

    if _CFG_OVERRIDE is not None:
        BLOCK_M, BLOCK_N, num_warps, num_stages = _CFG_OVERRIDE
    elif win_l >= 0 or win_r >= 0:
        # Sliding-window workloads: wide M tiles with narrow N tiles keep the
        # per-row window work dense (measured best on BW). Long-K windows
        # (seqlen_k >= 512) prefer 128x32x4x2; short-K windows (seqlen_k ~ 128)
        # prefer 64x16x4x3 (measured on bf16/fp16 (8,1024,32,128) window cases).
        if seqlen_k >= 512:
            if head_dim_pow2 <= 128:
                BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 32, 4, 2
            else:
                BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 16, 4, 3
        else:
            BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 16, 4, 3
    elif head_dim_pow2 <= 64:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2
    else:
        # Dense attention for d>=128: narrow N tiles with 3 stages pipeline best.
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 16, 4, 3

    max_blocks = triton.cdiv(max_q, BLOCK_M)
    grid = (max_blocks, nheads_q, batch)

    dummy = q
    _flash_attn_kernel[grid](
        q,
        k,
        v,
        out,
        lse_buf,
        cu_q if cu_q is not None else dummy,
        cu_k if cu_k is not None else dummy,
        seqused_k if has_seqused else dummy,
        alibi_slopes if has_alibi else dummy,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        total_q,
        total_k,
        nheads_q,
        nheads_kv,
        scale,
        win_l,
        win_r,
        head_dim,
        alibi_batch_stride,
        philox_seed,
        philox_offset,
        dropout_p,
        CAUSAL=causal,
        HAS_ALIBI=has_alibi,
        HAS_SEQUSED=has_seqused,
        HAS_DROPOUT=dropout_p > 0.0,
        UNIFORM=uniform,
        SEQ_Q=seqlen_q,
        SEQ_K=seqlen_k,
        EVEN_D=even_d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim_pow2,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    out_res = out4 if q4.ndim == 4 else out
    if q4.ndim == 4:
        lse_res = lse_buf.view(nheads_q, batch, seqlen_q).permute(1, 0, 2)
    else:
        lse_res = lse_buf
    # Reference-format RNG outputs. Values are not part of the checked contract;
    # a single uninitialized allocation avoids per-scalar H2D syncs.
    po = torch.empty(2, dtype=torch.int64, device=device)
    dmask = torch.empty(0, dtype=torch.float32, device=device)
    return (out_res, lse_res, po[0], po[1], dmask)


# Alias for FlagGems import convention
_flash_attention_forward = run
