"""Kunlunxin XPU3 override of `aten::_scaled_dot_product_attention_math`.

Why the generic kernel cannot be used here
------------------------------------------
* It feeds masked 2-D loads straight into `tl.dot`, which fails to lower in
  `ConvertTritonSDNNToLLVM` (wiki
  ``patterns/xpu3-dot-masked-load-miscompile.md``).
* It uses a float select on the softmax denominator
  (`tl.where(l == 0.0, 1.0, l)`), which the sibling `_for_mps` generic kernel
  reports as `LLVM ERROR: Invalid selectType_: cannot invert non-select EW
  funct`.
* Even after removing those, XPU3 miscompiles a `tl.dot` whose operand is a
  *register-derived* tile: the PV product `dot(exp2(dot(q, k^T)), v)` came back
  with order-1 relative error for fp16/bf16/fp32 in a standalone probe, while
  the QK dot itself and every `tl.max`/`tl.sum` on it were exact.  Only
  load-rooted dot operands are reliable.
* A single kernel that keeps both the softmax-statistics loop and the
  weight-materialisation loop was also miscompiled (measured), so the passes
  are split; each kernel below was validated in isolation.

Design (three kernels)
----------------------
1. ``_sdpa_stats_kernel``  — QK scores → per-row softmax stats (m, l).
2. ``_sdpa_weights_kernel`` — recompute QK, normalise by the loaded stats and
   write the full attention-weight matrix (the required output).
3. ``_sdpa_pv_kernel`` — ``out = weights @ V`` as a tiled GEMM whose operands
   are both loads.

Every `tl.dot` operand is an unmasked, affine, in-bounds load.  Partial tiles
are handled by clamping the block *base address* (a scalar) so the tail block
overlaps instead of masking lanes (per-lane `tl.minimum` inside a dot kernel
trips the SDNN i1-DMA lowering:
``patterns/xpu3-clamp-in-dot-kernel-bf16.md``).  Masks are applied with
`tl.where` to the *dot result*, never to an operand.  Both dots are 2-arg with
an explicit accumulator; the denominator is ``1 / l`` with no select.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _sdpa_stats_kernel(
    Q,
    K,
    ATTN_MASK,
    M,
    L,
    sm_scale,
    CAUSAL_DIAG,
    stride_qz: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_mz: tl.constexpr,
    stride_mm: tl.constexpr,
    stride_mn: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_M: tl.constexpr,
    EXACT_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    # Scalar base clamp: the last block is shifted back so it reads real rows
    # (overlapping the previous block) instead of out-of-range lanes.
    if EXACT_M:
        m_base = start_m * BLOCK_M
    else:
        m_base = tl.minimum(start_m * BLOCK_M, Q_CTX - BLOCK_M)
    offs_m = m_base + tl.arange(0, BLOCK_M)

    if BLOCK_D == HEAD_DIM:
        rd = tl.arange(0, BLOCK_D)
    else:
        # Non-power-of-two head dims keep the addresses in-bounds and zero the
        # padding lanes by arithmetic (a per-lane select on a dot operand is
        # miscompiled on this backend).  Attention test matrices only use
        # power-of-two head sizes, so this branch is not exercised.
        rd = tl.minimum(tl.arange(0, BLOCK_D), HEAD_DIM - 1)

    query = tl.load(
        Q + off_z * stride_qz + offs_m[:, None] * stride_qm + rd[None, :] * stride_qk
    )
    if BLOCK_D != HEAD_DIM:
        dkeep = (tl.arange(0, BLOCK_D) < HEAD_DIM).to(query.dtype)
        query = query * dkeep[None, :]

    LOG2E: tl.constexpr = 1.4426950408889634
    NEG = -1.0e30
    diag = CAUSAL_DIAG

    # -1e30 instead of -inf keeps `m_i - m_new` finite for never-touched rows.
    m_i = tl.full([BLOCK_M], NEG, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, KV_CTX, BLOCK_N):
        if EXACT_N:
            n_base = start_n
        else:
            n_base = tl.minimum(start_n, KV_CTX - BLOCK_N)
        offs_n = n_base + tl.arange(0, BLOCK_N)
        # An overlapping tail block re-reads columns below `start_n`; only its
        # own lanes may contribute, otherwise they would be double counted.
        owned = offs_n >= start_n

        key = tl.load(
            K
            + off_z * stride_kz
            + offs_n[:, None] * stride_kn
            + rd[None, :] * stride_kk
        )
        if BLOCK_D != HEAD_DIM:
            key = key * dkeep[None, :]

        qk = tl.dot(query, tl.trans(key), out_dtype=tl.float32, allow_tf32=False)
        qk = qk * sm_scale

        if HAS_ATTN_MASK:
            m_ptrs = (
                ATTN_MASK
                + off_z * stride_mz
                + offs_m[:, None] * stride_mm
                + offs_n[None, :] * stride_mn
            )
            qk = qk + tl.load(m_ptrs).to(tl.float32)

        if EXACT_N and not IS_CAUSAL:
            pass  # every lane of a full, owned block is a real key column
        elif EXACT_N:
            qk = tl.where(offs_n[None, :] <= offs_m[:, None] + diag, qk, NEG)
        else:
            keep = owned[None, :]
            if IS_CAUSAL:
                keep = keep & (offs_n[None, :] <= offs_m[:, None] + diag)
            qk = tl.where(keep, qk, NEG)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2((m_i - m_new) * LOG2E)
        p = tl.exp2((qk - m_new[:, None]) * LOG2E)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    tl.store(M + off_z * Q_CTX + offs_m, m_i)
    tl.store(L + off_z * Q_CTX + offs_m, l_i)


@libentry()
@triton.jit
def _sdpa_weights_kernel(
    Q,
    K,
    ATTN_MASK,
    M,
    L,
    Weights,
    sm_scale,
    CAUSAL_DIAG,
    stride_qz: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_mz: tl.constexpr,
    stride_mm: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_wz: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_M: tl.constexpr,
    EXACT_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    if EXACT_M:
        m_base = start_m * BLOCK_M
    else:
        m_base = tl.minimum(start_m * BLOCK_M, Q_CTX - BLOCK_M)
    offs_m = m_base + tl.arange(0, BLOCK_M)

    if BLOCK_D == HEAD_DIM:
        rd = tl.arange(0, BLOCK_D)
    else:
        rd = tl.minimum(tl.arange(0, BLOCK_D), HEAD_DIM - 1)

    query = tl.load(
        Q + off_z * stride_qz + offs_m[:, None] * stride_qm + rd[None, :] * stride_qk
    )
    if BLOCK_D != HEAD_DIM:
        dkeep = (tl.arange(0, BLOCK_D) < HEAD_DIM).to(query.dtype)
        query = query * dkeep[None, :]

    m_i = tl.load(M + off_z * Q_CTX + offs_m)
    # Every row owns at least one key column (a causal row always keeps column
    # 0), so the online sum is >= 1 and the reciprocal is well defined.  Using
    # it directly also avoids a float select / max on the denominator, which
    # this backend mis-lowers.
    inv_l = 1.0 / tl.load(L + off_z * Q_CTX + offs_m)

    LOG2E: tl.constexpr = 1.4426950408889634
    diag = CAUSAL_DIAG
    n_blocks = tl.cdiv(KV_CTX, BLOCK_N)

    # Reverse block order: an overlapping tail block writes zeros to the columns
    # it does not own, and those are (re)written by the block that owns them
    # later in this loop.
    for idx in range(n_blocks - 1, -1, -1):
        start_n = idx * BLOCK_N
        if EXACT_N:
            n_base = start_n
        else:
            n_base = tl.minimum(start_n, KV_CTX - BLOCK_N)
        offs_n = n_base + tl.arange(0, BLOCK_N)

        key = tl.load(
            K
            + off_z * stride_kz
            + offs_n[:, None] * stride_kn
            + rd[None, :] * stride_kk
        )
        if BLOCK_D != HEAD_DIM:
            key = key * dkeep[None, :]

        qk = tl.dot(query, tl.trans(key), out_dtype=tl.float32, allow_tf32=False)
        qk = qk * sm_scale

        if HAS_ATTN_MASK:
            m_ptrs = (
                ATTN_MASK
                + off_z * stride_mz
                + offs_m[:, None] * stride_mm
                + offs_n[None, :] * stride_mn
            )
            qk = qk + tl.load(m_ptrs).to(tl.float32)

        p = tl.exp2((qk - m_i[:, None]) * LOG2E)
        if EXACT_N and not IS_CAUSAL:
            pass
        elif EXACT_N:
            p = tl.where(offs_n[None, :] <= offs_m[:, None] + diag, p, 0.0)
        else:
            keep = offs_n[None, :] >= start_n
            if IS_CAUSAL:
                keep = keep & (offs_n[None, :] <= offs_m[:, None] + diag)
            p = tl.where(keep, p, 0.0)
        p = p * inv_l[:, None]

        tl.store(
            Weights
            + off_z * stride_wz
            + offs_m[:, None] * stride_wm
            + offs_n[None, :] * stride_wn,
            p.to(Weights.dtype.element_ty),
        )


@libentry()
@triton.jit
def _sdpa_pv_kernel(
    Weights,
    V,
    Out,
    stride_wz: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_M: tl.constexpr,
):
    """out[m, :] = sum_n weights[m, n] * v[n, :] — both dot operands are loads."""
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    if EXACT_M:
        m_base = start_m * BLOCK_M
    else:
        m_base = tl.minimum(start_m * BLOCK_M, Q_CTX - BLOCK_M)
    offs_m = m_base + tl.arange(0, BLOCK_M)

    if BLOCK_DV == VALUE_DIM:
        rdv = tl.arange(0, BLOCK_DV)
    else:
        rdv = tl.minimum(tl.arange(0, BLOCK_DV), VALUE_DIM - 1)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    # BLOCK_N divides KV_CTX (chosen on the host), so every tile here is a
    # full, in-bounds affine load -- no clamp and no mask on the dot operands.
    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        w_tile = tl.load(
            Weights
            + off_z * stride_wz
            + offs_m[:, None] * stride_wm
            + offs_n[None, :] * stride_wn
        )
        v_tile = tl.load(
            V
            + off_z * stride_vz
            + offs_n[:, None] * stride_vn
            + rdv[None, :] * stride_vk
        )
        acc += tl.dot(w_tile, v_tile, out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        Out
        + off_z * stride_oz
        + offs_m[:, None] * stride_om
        + rdv[None, :] * stride_ok,
        acc.to(Out.dtype.element_ty),
    )


def _pick_block(n, cap):
    """Largest power of two <= n, capped at `cap` (block sizes divide the
    sequence lengths whenever possible so tail blocks do not overlap)."""
    block = 1
    while block * 2 <= n:
        block *= 2
    return min(cap, block)


def _pick_divisor(n, cap):
    """Largest power of two that divides `n`, capped at `cap`."""
    block = 1
    while block * 2 <= cap and n % (block * 2) == 0:
        block *= 2
    return block


def _sdpa_math_impl(
    query,
    key,
    value,
    attn_mask=None,
    is_causal=False,
    scale=None,
    causal_diag=None,
):
    """Shared host path for the math and math_for_mps overrides."""
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_ATTENTION_MATH")

    assert query.dim() == 4, "expected 4D (batch, heads, seq_len, head_dim) tensors"

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, heads, q_seq_len, head_dim = query.shape
    kv_seq_len = key.shape[2]
    value_dim = value.shape[3]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    else:
        scale = float(scale)

    if causal_diag is None:
        causal_diag = 0

    assert q_seq_len >= 8 and kv_seq_len >= 8, "sequence lengths must be >= 8"

    bh = batch * heads
    q2 = query.view(bh, q_seq_len, head_dim)
    k2 = key.view(bh, kv_seq_len, head_dim)
    v2 = value.view(bh, kv_seq_len, value_dim)

    # All stores are in-bounds by construction (base-clamped blocks and
    # divide-exact PV tiles), so plain contiguous buffers are enough.
    out = torch.empty_like(query)
    o2 = out.view(bh, q_seq_len, value_dim)
    weights = torch.empty(
        (bh, q_seq_len, kv_seq_len), dtype=query.dtype, device=query.device
    )
    stats_m = torch.empty((bh, q_seq_len), dtype=torch.float32, device=query.device)
    stats_l = torch.empty((bh, q_seq_len), dtype=torch.float32, device=query.device)

    has_attn_mask = attn_mask is not None
    if has_attn_mask:
        assert attn_mask.dtype != torch.bool, "boolean attn_mask is not supported"
        am = attn_mask.expand(batch, heads, q_seq_len, kv_seq_len)
        # The kernel folds (batch, head) into one grid axis, so the per-group
        # stride is dim 1; dim 3 is the key stride.  A broadcast mask yields 0
        # for the folded strides, which is exactly what we want.
        m_strides = (am.stride(1), am.stride(2), am.stride(3))
        mask_buf = am
    else:
        mask_buf = query  # unused; HAS_ATTN_MASK guards every access
        m_strides = (0, 0, 0)

    BLOCK_M = _pick_block(q_seq_len, 64)
    BLOCK_N = _pick_block(kv_seq_len, 64)
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_DV = triton.next_power_of_2(value_dim)
    # The PV GEMM cannot tolerate an overlapping tail tile (it would double
    # count columns), so its key tile must divide KV_CTX exactly.
    BLOCK_N_PV = _pick_divisor(kv_seq_len, 64)

    grid = (triton.cdiv(q_seq_len, BLOCK_M), bh)

    with torch_device_fn.device(query.device):
        _sdpa_stats_kernel[grid](
            q2,
            k2,
            mask_buf,
            stats_m,
            stats_l,
            scale,
            causal_diag,
            q2.stride(0),
            q2.stride(1),
            q2.stride(2),
            k2.stride(0),
            k2.stride(1),
            k2.stride(2),
            m_strides[0],
            m_strides[1],
            m_strides[2],
            q_seq_len,
            kv_seq_len,
            HEAD_DIM=head_dim,
            BLOCK_D=BLOCK_D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EXACT_M=q_seq_len % BLOCK_M == 0,
            EXACT_N=kv_seq_len % BLOCK_N == 0,
            IS_CAUSAL=is_causal,
            HAS_ATTN_MASK=has_attn_mask,
        )
        _sdpa_weights_kernel[grid](
            q2,
            k2,
            mask_buf,
            stats_m,
            stats_l,
            weights,
            scale,
            causal_diag,
            q2.stride(0),
            q2.stride(1),
            q2.stride(2),
            k2.stride(0),
            k2.stride(1),
            k2.stride(2),
            m_strides[0],
            m_strides[1],
            m_strides[2],
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            q_seq_len,
            kv_seq_len,
            HEAD_DIM=head_dim,
            BLOCK_D=BLOCK_D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EXACT_M=q_seq_len % BLOCK_M == 0,
            EXACT_N=kv_seq_len % BLOCK_N == 0,
            IS_CAUSAL=is_causal,
            HAS_ATTN_MASK=has_attn_mask,
        )
        _sdpa_pv_kernel[grid](
            weights,
            v2,
            o2,
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            v2.stride(0),
            v2.stride(1),
            v2.stride(2),
            o2.stride(0),
            o2.stride(1),
            o2.stride(2),
            q_seq_len,
            kv_seq_len,
            VALUE_DIM=value_dim,
            BLOCK_DV=BLOCK_DV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N_PV,
            EXACT_M=q_seq_len % BLOCK_M == 0,
        )

    return out, weights.view(batch, heads, q_seq_len, kv_seq_len)


def _scaled_dot_product_attention_math(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """Kunlunxin override: math SDPA returning (output, attention weights)."""
    assert dropout_p == 0.0, "dropout_p != 0.0 is not supported"
    return _sdpa_math_impl(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )
