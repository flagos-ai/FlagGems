"""
attention.py — ARM CPU Flash Attention (Triton-CPU)

Flash Attention v2 with online softmax — no O(M*N) intermediate matrix.
Supports GQA (grouped-query attention), is_causal, BF16 inputs.

Performance (M=512, D=128, H=16, OMP=6, CIX P1 CD8180):
  ATen:  ~179ms  ->  Triton:  ~40ms  (4.5x speedup)
  BLOCK_M=32, BLOCK_N=16 chosen via sweep.

Decode path (M < BLOCK_M=32) falls back to ATen: tl.dot requires M>=4.
Non-BF16 inputs or attn_mask also fall back to ATen.
"""
import ctypes
import logging
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

log = logging.getLogger(__name__)

# Preload libsleef (tl.math.exp2 in Triton-CPU .so depends on SLEEF symbols).

def _ensure_sleef():
    try:
        import triton as _t
        sleef_dir = os.path.join(os.path.dirname(_t.__file__), "_C")
        sleef_so  = os.path.join(sleef_dir, "libsleef.so.3")
        if not os.path.exists(sleef_so):
            return
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if sleef_dir not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{sleef_dir}:{ld}"
        ctypes.CDLL(sleef_so)           # preload so later dlopen can resolve symbols
    except Exception:
        pass

_ensure_sleef()

# Keep the original ATen SDPA for internal fallback (avoids infinite recursion after monkey-patch).
_aten_sdpa = F.scaled_dot_product_attention

# Import once at module load. If triton-cpu lacks the runtime module (older
# build), fall through to ATen for M=1 decode.
try:
    from triton.language.extra.cpu.runtime import flash_attn_decode_bf16 as _flash_attn_decode_bf16
except ImportError:
    _flash_attn_decode_bf16 = None

# log2(e) = 1/ln(2) — used so we can substitute exp2 for exp (avoids SLEEF precision loss).
_LOG2E: float = 1.44269504089

# Block sizes (BLOCK_N=16 chosen via sweep).
_BLOCK_M: int = 32
_BLOCK_N: int = 16

# ── Flash Attention Triton Kernel ───────────────────────────────────────────

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale, Out,
    # [B*Hq, M, D]
    stride_qh, stride_qm, stride_qk,
    # [B*Hkv, N, D]
    stride_kh, stride_kn, stride_kk,
    # [B*Hkv, N, D]
    stride_vh, stride_vn, stride_vk,
    # [B*Hq, M, D]
    stride_oh, stride_om, stride_ok,
    seqlen_q, seqlen_k,
    q_numhead, kv_numhead,          # GQA support
    LOG2E: tl.constexpr,            # 1.44269504
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,        # compile-time constant: generates two code paths
):
    pid_bh = tl.program_id(0)       # batch * Q-head (flattened)
    pid_m  = tl.program_id(1)       # M-tile index

    # GQA mapping: every (Hq // Hkv) Q-heads share one KV-head.
    head_id    = pid_bh % q_numhead
    batch_id   = pid_bh // q_numhead
    kv_head_id = head_id * kv_numhead // q_numhead

    Q_bh = Q   + (batch_id * q_numhead  + head_id)    * stride_qh
    K_bh = K   + (batch_id * kv_numhead + kv_head_id) * stride_kh
    V_bh = V   + (batch_id * kv_numhead + kv_head_id) * stride_vh
    O_bh = Out + (batch_id * q_numhead  + head_id)    * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q

    # Q: [BLOCK_M, HEAD_DIM], pre-multiplied by sm_scale*LOG2E (shifts into log2 domain).
    q = tl.load(
        Q_bh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=mask_m[:, None], other=0.0,
    ).to(tl.float32) * (sm_scale * LOG2E)

    # Online softmax state (per-row, log2 domain).
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    lse = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal: only iterate up to the current Q-tile.
    if IS_CAUSAL:
        kv_end = tl.minimum(seqlen_k, (pid_m + 1) * BLOCK_M)
    else:
        kv_end = seqlen_k

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen_k

        # K^T: [HEAD_DIM, BLOCK_N] — swapping k/n offsets gives the transposed load.
        k = tl.load(
            K_bh + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn,
            mask=mask_n[None, :], other=0.0,
        ).to(tl.float32)

        # QK^T: [BLOCK_M, HEAD_DIM] x [HEAD_DIM, BLOCK_N] -> [BLOCK_M, BLOCK_N].
        # q is already in log2 domain (includes sm_scale*LOG2E), so exp2 can be applied directly.
        qk = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16)).to(tl.float32)

        if IS_CAUSAL:
            causal_ok = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_ok & mask_n[None, :], qk, float('-inf'))
        else:
            qk = tl.where(mask_n[None, :], qk, float('-inf'))

        # Online softmax (log2 domain).
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))       # [BLOCK_M]
        alpha  = tl.math.exp2(m_i - m_new)                  # rescale previous rows
        p      = tl.math.exp2(qk - m_new[:, None])         # [BLOCK_M, BLOCK_N]

        lse = lse * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V: [BLOCK_N, HEAD_DIM]
        v = tl.load(
            V_bh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.bfloat16)

        # P @ V: [BLOCK_M, BLOCK_N] × [BLOCK_N, HEAD_DIM] → [BLOCK_M, HEAD_DIM]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        m_i = m_new

    # Normalize and write back.
    acc = acc / lse[:, None]
    tl.store(
        O_bh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None],
    )


# Python wrappers.

def _triton_flash_attn(
    query: torch.Tensor,
    key:   torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Core Triton kernel invocation. Caller must have already verified the Triton path applies."""
    B, Hq, M, D = query.shape
    Hkv = key.shape[1]

    # Flatten batch+head -> [B*H, seq, D]
    q   = query.reshape(B * Hq,  M, D)
    k   = key.reshape(B * Hkv, -1, D)
    v   = value.reshape(B * Hkv, -1, D)
    N   = k.shape[1]
    out = torch.empty_like(q)

    grid = (B * Hq, triton.cdiv(M, _BLOCK_M))

    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        M, N,
        Hq, Hkv,
        _LOG2E,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=is_causal,
    )
    return out.reshape(B, Hq, M, D)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """
    aten::scaled_dot_product_attention — ARM CPU Flash Attention.

    Triton path conditions (otherwise fall back to ATen):
      - dtype = bfloat16
      - attn_mask = None
      - dropout_p = 0.0
      - seqlen_q >= BLOCK_M (=32)
      - head_dim in {16,32,64,128,256}
    """
    B, Hq, M, D = query.shape

    # M=1 decode fast path: C runtime flash_attn_decode_bf16 via triton-cpu.
    # Measured +1.2% E2E on Qwen3-1.7B INT8 vs ATen fallback (3 rounds A/B).
    # Requires BF16, no mask, no dropout, contiguous Q/K/V.
    if (_flash_attn_decode_bf16 is not None
            and M == 1 and B == 1
            and query.dtype == torch.bfloat16
            and attn_mask is None
            and dropout_p == 0.0
            and query.is_contiguous()
            and key.is_contiguous() and value.is_contiguous()):
        Hkv = key.shape[1]
        seq_len = key.shape[2]
        sm_scale = scale if scale is not None else D ** -0.5
        q_flat = query.squeeze(0).squeeze(1).contiguous()
        k_flat = key.squeeze(0).contiguous()
        v_flat = value.squeeze(0).contiguous()
        out_flat = torch.empty(Hq, D, dtype=torch.bfloat16)
        _flash_attn_decode_bf16(
            q_flat, k_flat, v_flat, out_flat,
            seq_len, D, sm_scale, Hq, Hkv,
            k_flat.stride(1), v_flat.stride(1),
        )
        return out_flat.unsqueeze(0).unsqueeze(2)

    # Prefill fast path: Triton Flash Attention kernel (requires M >= BLOCK_M).
    use_triton = (
        query.dtype == torch.bfloat16
        and attn_mask is None
        and dropout_p == 0.0
        and M >= _BLOCK_M
        and D in {16, 32, 64, 128, 256}
    )

    if not use_triton:
        log.debug("GEMS SDPA: ATen fallback (M=%d, dtype=%s, mask=%s)",
                  M, query.dtype, attn_mask is not None)
        return _aten_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    sm_scale = scale if scale is not None else D ** -0.5
    log.debug("GEMS SDPA: Triton Flash Attention (M=%d, N=%d, D=%d, causal=%s, Hq=%d, Hkv=%d)",
              M, key.shape[2], D, is_causal, Hq, key.shape[1])
    return _triton_flash_attn(query, key, value, sm_scale, is_causal)
