import math

import torch
import triton
import triton.language as tl


@triton.jit
def _scaled_dot_product_attention_math_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    attn_ptr, has_attn_mask, dropout_p, rng_seed,
    B, H, M, N, D, DV,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_mb, stride_mh, stride_mm, stride_mn,
    is_causal, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    b = pid_bh // H
    h = pid_bh % H

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Bases
    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    o_base = out_ptr + b * stride_ob + h * stride_oh
    m_base = attn_ptr + b * stride_mb + h * stride_mh

    # Online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Precompute dropout constants
    inv_keep_prob = tl.where(dropout_p > 0, 1.0 / (1.0 - dropout_p), 1.0)

    # Iterate over K/V sequence in blocks
    n_start = 0
    while n_start < N:
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # Compute logits block: scores = Q @ K^T for this tile
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        d_start = 0
        while d_start < D:
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < D

            # Load Q sub-block [M, Dsub]
            q_ptrs = q_base + (m_offsets[:, None] * stride_qm) + (
                d_offsets[None, :] * stride_qd
            )
            q_sub = tl.load(
                q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0
            ).to(tl.float32)

            # Load K^T sub-block [Dsub, Nblock] — transposed load avoids explicit tl.trans
            kt_ptrs = k_base + (d_offsets[:, None] * stride_kd) + (
                n_offsets[None, :] * stride_kn
            )
            kt_sub = tl.load(
                kt_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0
            ).to(tl.float32)

            scores += tl.dot(q_sub, kt_sub)

            d_start += BLOCK_D

        # Apply scale
        scores = scores * scale

        # Apply additive attention mask if provided: shape [B, H, M, N]
        if has_attn_mask != 0:
            mask_ptrs = m_base + (m_offsets[:, None] * stride_mm) + (
                n_offsets[None, :] * stride_mn
            )
            mask_vals = tl.load(
                mask_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0
            ).to(tl.float32)
            scores = scores + mask_vals

        # Apply causal mask if requested
        if is_causal != 0:
            mm = m_offsets[:, None]
            nn = n_offsets[None, :]
            causal_keep = nn <= mm
            scores = tl.where(causal_keep, scores, -float("inf"))

        # Invalidate out-of-bounds columns
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        # Compute online softmax update
        row_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, row_max)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])
        l_part = tl.sum(p, axis=1)
        l_i_new = l_i * alpha + l_part

        coeff_prev = tl.where(l_i_new > 0, (l_i * alpha) / l_i_new, 0.0)
        inv_l_new = tl.where(l_i_new > 0, 1.0 / l_i_new, 0.0)

        # Apply dropout (after softmax)
        if dropout_p > 0:
            bh = b * H + h
            base_offset = (bh * M) * N
            rand_offsets = base_offset + (m_offsets[:, None] * N) + n_offsets[None, :]
            r = tl.rand(rng_seed, rand_offsets)
            keep = r > dropout_p
            drop_factor = tl.where(keep, inv_keep_prob, 0.0)
            p_drop = p * drop_factor
        else:
            p_drop = p

        # Update accumulated output across DV in blocks
        dv_start = 0
        while dv_start < DV:
            dv_offsets = dv_start + tl.arange(0, BLOCK_DV)
            dv_mask = dv_offsets < DV

            # Load V subtile [Nblock, DVsub]
            v_ptrs = v_base + (n_offsets[:, None] * stride_vn) + (
                dv_offsets[None, :] * stride_vd
            )
            v_sub = tl.load(
                v_ptrs, mask=n_mask[:, None] & dv_mask[None, :], other=0.0
            ).to(tl.float32)

            # Contribution from current block
            contrib = tl.dot(p_drop, v_sub)  # [M, DVsub]
            contrib = contrib * inv_l_new[:, None]

            # Read old acc, scale by coeff_prev, add contrib, write back
            o_ptrs = o_base + (m_offsets[:, None] * stride_om) + (
                dv_offsets[None, :] * stride_od
            )
            old_acc = tl.load(
                o_ptrs, mask=m_mask[:, None] & dv_mask[None, :], other=0.0
            ).to(tl.float32)
            new_acc = old_acc * coeff_prev[:, None] + contrib

            tl.store(o_ptrs, new_acc, mask=m_mask[:, None] & dv_mask[None, :])

            dv_start += BLOCK_DV

        # Update running softmax stats
        m_i = m_i_new
        l_i = l_i_new

        n_start += BLOCK_N


def _scaled_dot_product_attention_math(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    """Scaled Dot-Product Attention (math backend) implemented in Triton.

    Computes Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + attn_mask) @ V
    using online softmax (tiling) to avoid materializing the full attention matrix.

    Args:
        q: Query tensor of shape (B, H, M, D)
        k: Key tensor of shape (B, H, N, D)
        v: Value tensor of shape (B, H, N, DV)
        attn_mask: Optional attention mask broadcastable to (B, H, M, N).
                   Bool masks (True=keep) or additive float masks are supported.
        dropout_p: Dropout probability (default 0.0)
        is_causal: Whether to apply causal mask (default False)
        scale: Scale factor (default 1/sqrt(D))

    Returns:
        Output tensor of shape (B, H, M, DV)
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, (
        "q, k, v must be 4D tensors (B, H, L, D)"
    )
    B, H, M, D = q.shape
    Bk, Hk, N, Dk = k.shape
    Bv, Hv, Nv, DV = v.shape
    assert B == Bk == Bv and H == Hk == Hv and N == Nv and D == Dk, (
        "Shape mismatch among q, k, v"
    )

    device = q.device
    q_dtype = q.dtype

    # Compute scale value
    if scale is None:
        scale_val = 1.0 / math.sqrt(D)
    else:
        if isinstance(scale, torch.Tensor):
            if scale.numel() == 1:
                scale_val = float(scale.item())
            else:
                raise NotImplementedError(
                    "scale tensor with numel>1 is not supported in this kernel"
                )
        else:
            scale_val = float(scale)

    # Prepare additive attention mask if provided
    has_attn_mask = 0
    attn_mask_tensor = torch.empty(1, device=device, dtype=torch.float32)
    if attn_mask is not None:
        has_attn_mask = 1
        if attn_mask.dtype == torch.bool:
            attn_mask_tensor = torch.where(
                attn_mask.to(device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(float("-inf"), device=device),
            ).to(torch.float32)
        else:
            attn_mask_tensor = attn_mask.to(device=device, dtype=torch.float32)
        # Broadcast to (B, H, M, N) if needed
        attn_mask_tensor = attn_mask_tensor.expand(B, H, M, N).contiguous()

    # Allocate output in float32 for numerical stability (zero-init for online accumulation)
    out_fp32 = torch.zeros((B, H, M, DV), device=device, dtype=torch.float32)

    # Get strides
    stride_qb, stride_qh, stride_qm, stride_qd = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = v.stride()
    stride_ob, stride_oh, stride_om, stride_od = out_fp32.stride()
    if has_attn_mask:
        stride_mb, stride_mh, stride_mm, stride_mn = attn_mask_tensor.stride()
    else:
        stride_mb = stride_mh = stride_mm = stride_mn = 0

    # Choose block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, D) if D <= 64 else (128 if D <= 128 else 256)
    BLOCK_DV = min(64, DV) if DV <= 64 else (128 if DV <= 128 else 256)

    # Ensure BLOCK_D >= 16 for tl.dot compatibility
    BLOCK_D = max(BLOCK_D, 16)
    BLOCK_DV = max(BLOCK_DV, 16)

    # Launch grid
    grid = (triton.cdiv(M, BLOCK_M), B * H)

    # RNG seed for dropout
    if dropout_p and dropout_p > 0.0:
        rng_seed = int(torch.randint(0, 2**31 - 1, (1,), device="cpu").item())
    else:
        rng_seed = 0

    _scaled_dot_product_attention_math_kernel[grid](
        q,
        k,
        v,
        out_fp32,
        attn_mask_tensor,
        has_attn_mask,
        float(dropout_p),
        rng_seed,
        B, H, M, N, D, DV,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_mb, stride_mh, stride_mm, stride_mn,
        int(is_causal),
        float(scale_val),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        num_warps=4,
        num_stages=2,
    )

    return out_fp32.to(dtype=q_dtype)
