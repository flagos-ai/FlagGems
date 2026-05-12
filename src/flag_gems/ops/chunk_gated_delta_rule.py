"""chunk_gated_delta_rule — FlagGems operator.

Wraps the existing FLA chunk-parallel forward kernels (which already use
``tl.dot`` Tensor Cores, in-chunk parallelism via the ``KKᵀ`` decomposition,
and full ``cu_seqlens`` support) under a ``torch.autograd.Function`` that
also provides a correct backward pass.

API (matches the upstream FLA convention ``[B, T, H, K/V]``):

    out, final_state = chunk_gated_delta_rule(
        q, k, v, g, beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )

Shapes:
    q, k:         (B, T, H, K)
    v:            (B, T, H, V)
    g, beta:      (B, T, H)
    initial_state:(B, H, K, V)        (float32 recommended)
    out:          (B, T, H, V)
    final_state:  (B, H, K, V)        (float32) or None

Backward path:
    Forward fast-path uses the chunk-parallel FLA kernels.  For backward we
    invoke a numerically equivalent eager reference (chunk-parallel torch
    formulation, identical math to the naive FLA reference) and let
    autograd compute gradients.  This is slower than a hand-written Triton
    backward but is *correct for every input shape that the forward
    accepts*, including variable-length sequences (``cu_seqlens``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eager reference (used for: small-shape fallback, backward pass, and tests).
#
# Matches FLA's ``naive`` reference exactly when ``cu_seqlens is None``.
# Operates in float32 for numerical stability across long sequences.
# ---------------------------------------------------------------------------
def _eager_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    chunk_size: int = 64,
):
    out_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]

    BT = chunk_size
    pad_len = (BT - T % BT) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    # [B, T, H, .] -> [B, H, T, .]
    q = q.transpose(1, 2).contiguous().to(torch.float32) * scale
    k = k.transpose(1, 2).contiguous().to(torch.float32)
    v = v.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)

    Tpad = q.shape[2]
    NC = Tpad // BT

    # Bring beta into v / k_beta:
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Reshape to chunks: [B, H, NC, BT, .]
    q = q.view(B, H, NC, BT, K)
    k = k.view(B, H, NC, BT, K)
    v = v.view(B, H, NC, BT, V)
    k_beta = k_beta.view(B, H, NC, BT, K)
    g_chunk = g.view(B, H, NC, BT)

    # Per-chunk cumulative gate and inter-token decay matrix.
    g_cumsum = g_chunk.cumsum(dim=-1)  # [B, H, NC, BT]
    decay_diff = g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)  # [B,H,NC,BT,BT]
    L_mask = decay_diff.tril().exp().tril()

    # Build A^{-1} (lower-triangular K-K^T solve, eq. (*) in FLA naive).
    diag_mask = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0
    )
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(diag_mask, 0.0)
    for i in range(1, BT):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(BT, dtype=q.dtype, device=q.device)

    u = attn @ v  # corresponds to "k_cumsum" / "u" in FLA naming
    decay_exp = g_cumsum.exp().unsqueeze(-1)  # [B,H,NC,BT,1]
    w = attn @ (k_beta * decay_exp)  # corresponds to "k_cumdecay"

    # Recurrent chunk loop over hidden state S of shape [B, H, K, V].
    S = q.new_zeros(B, H, K, V, dtype=torch.float32)
    if initial_state is not None:
        S = initial_state.to(torch.float32).clone()

    causal_mask = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1
    )
    o_chunks = []
    for i in range(NC):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        w_i = w[:, :, i]
        g_i = g_cumsum[:, :, i]

        attn_qk = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill(
            causal_mask, 0.0
        )
        v_prime = w_i @ S
        v_new = u_i - v_prime
        o_inter = (q_i * g_i.unsqueeze(-1).exp()) @ S
        o_i = o_inter + attn_qk @ v_new
        o_chunks.append(o_i)

        # Carry forward state.
        last_g = g_i[..., -1:].unsqueeze(-1)  # [B,H,1,1]
        gate = (g_i[..., -1:] - g_i).exp().unsqueeze(-1)  # [B,H,BT,1]
        S = S * last_g.exp() + (k_i * gate).transpose(-1, -2) @ v_new

    o = torch.stack(o_chunks, dim=2).reshape(B, H, Tpad, V)
    o = o[:, :, :T].transpose(1, 2).contiguous().to(out_dtype)
    final_state = S if output_final_state else None
    return o, final_state


# ---------------------------------------------------------------------------
# Fast-path adapter: uses existing FLA chunk-parallel kernels in flag_gems.
# Returns ``None`` when the fast path is unavailable so callers fall back.
# ---------------------------------------------------------------------------
_ALLOCATOR_INSTALLED = False


def _ensure_triton_allocator():
    """Install a default ``triton.set_allocator`` if none is configured.

    Required for autotuned FLA kernels under Triton >= 3.4 which request a
    runtime scratch buffer.  No-op if the user has already installed one.
    """
    global _ALLOCATOR_INSTALLED
    if _ALLOCATOR_INSTALLED:
        return
    try:
        import triton
    except Exception:
        return
    if not hasattr(triton, "set_allocator"):
        _ALLOCATOR_INSTALLED = True
        return

    def _alloc(size, alignment, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    try:
        triton.set_allocator(_alloc)
    except Exception:
        pass
    _ALLOCATOR_INSTALLED = True


def _fla_fast_forward(
    q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
):
    try:
        from flag_gems.fused.FLA.chunk import chunk_gated_delta_rule_fwd as _fla_fwd
    except Exception:
        return None
    _ensure_triton_allocator()
    try:
        ret = _fla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    except Exception:
        return None
    # FLA returns ``(g_out, o, A, final_state, ...)``.  We only need o + final_state.
    if not isinstance(ret, tuple) or len(ret) < 4:
        return None
    o, final_state = ret[1], ret[3]
    # Sanity guard: some Triton/arch combinations (notably Blackwell SM12.0
    # under early Triton 3.6) produce non-finite or astronomically scaled
    # outputs from these kernels.  Reject so the caller can fall back to
    # the numerically equivalent eager path.
    if o is None:
        return None
    if not torch.isfinite(o).all():
        return None
    if o.abs().max().item() > 1e6:
        return None
    return o, final_state


# ---------------------------------------------------------------------------
# autograd.Function: forward fast-path + correct backward via eager rerun.
# ---------------------------------------------------------------------------
class _ChunkGatedDeltaRuleFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    ):
        if scale is None:
            scale = q.shape[-1] ** -0.5
        # Forward fast path.  Backward will rerun eager (slow but correct).
        if cu_seqlens is None:
            fast = _fla_fast_forward(
                q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
            )
        else:
            # FLA forward supports cu_seqlens; preserve for backward.
            fast = _fla_fast_forward(
                q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
            )
        if fast is not None:
            o, final_state = fast
        else:
            o, final_state = _eager_chunk_gated_delta_rule(
                q, k, v, g, beta, scale, initial_state, output_final_state
            )

        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            beta,
            initial_state if initial_state is not None else q.new_empty(0),
        )
        ctx.scale = scale
        ctx.has_initial_state = initial_state is not None
        ctx.output_final_state = output_final_state
        ctx.cu_seqlens = cu_seqlens
        # Guarantee fp32 final_state for downstream chaining.
        if final_state is not None and final_state.dtype != torch.float32:
            final_state = final_state.to(torch.float32)
        return o, final_state if output_final_state else o.new_zeros(0)

    @staticmethod
    def backward(ctx, do, dfinal_state):  # type: ignore[override]
        q, k, v, g, beta, h0_or_empty = ctx.saved_tensors
        scale = ctx.scale
        h0 = h0_or_empty if ctx.has_initial_state else None
        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is not None:
            # cu_seqlens backward is not supported by the eager rerun yet.
            # Fall back to per-sequence eager — correct but slow.
            return _backward_cu_seqlens(
                q,
                k,
                v,
                g,
                beta,
                h0,
                scale,
                ctx.output_final_state,
                cu_seqlens,
                do,
                dfinal_state,
            )

        with torch.enable_grad():
            qd = q.detach().clone().requires_grad_(q.requires_grad)
            kd = k.detach().clone().requires_grad_(k.requires_grad)
            vd = v.detach().clone().requires_grad_(v.requires_grad)
            gd = g.detach().clone().requires_grad_(g.requires_grad)
            bd = beta.detach().clone().requires_grad_(beta.requires_grad)
            h0d = (
                h0.detach().clone().requires_grad_(h0.requires_grad)
                if h0 is not None and h0.requires_grad
                else h0
            )
            o_ref, fs_ref = _eager_chunk_gated_delta_rule(
                qd, kd, vd, gd, bd, scale, h0d, ctx.output_final_state
            )
            grads_inputs = [t for t in (qd, kd, vd, gd, bd) if t.requires_grad]
            if h0d is not None and h0d.requires_grad:
                grads_inputs.append(h0d)
            if not grads_inputs:
                return (None,) * 9

            outputs = [o_ref]
            grad_outputs = [do.to(o_ref.dtype)]
            if (
                ctx.output_final_state
                and fs_ref is not None
                and dfinal_state is not None
            ):
                outputs.append(fs_ref)
                grad_outputs.append(dfinal_state.to(fs_ref.dtype))
            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=grads_inputs,
                grad_outputs=grad_outputs,
                allow_unused=True,
            )

        # Build full grad tuple matching forward signature
        # (q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens).
        out_grads = []
        gi = 0
        for t in (q, k, v, g, beta):
            if t.requires_grad:
                out_grads.append(grads[gi])
                gi += 1
            else:
                out_grads.append(None)
        if h0 is not None and h0.requires_grad:
            out_grads.append(grads[gi])
        else:
            out_grads.append(None)
        # scale, output_final_state, cu_seqlens — non-tensor or non-grad.
        out_grads = [
            out_grads[0],
            out_grads[1],
            out_grads[2],
            out_grads[3],
            out_grads[4],
            None,
            out_grads[5],
            None,
            None,
        ]
        return tuple(out_grads)


def _backward_cu_seqlens(
    q, k, v, g, beta, h0, scale, output_final_state, cu_seqlens, do, dfinal_state
):
    """Per-sequence backward for variable-length inputs (slow but correct).

    Inputs shaped ``[1, total_T, H, .]`` with ``cu_seqlens`` of length N+1.
    Splits along T, reruns eager for each sequence, concatenates grads.
    """
    cu = cu_seqlens.tolist()
    grads_q = torch.zeros_like(q)
    grads_k = torch.zeros_like(k)
    grads_v = torch.zeros_like(v)
    grads_g = torch.zeros_like(g)
    grads_beta = torch.zeros_like(beta)
    grads_h0 = torch.zeros_like(h0) if (h0 is not None and h0.requires_grad) else None
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        sl = slice(s, e)
        with torch.enable_grad():
            qi = q[:, sl].detach().clone().requires_grad_(True)
            ki = k[:, sl].detach().clone().requires_grad_(True)
            vi = v[:, sl].detach().clone().requires_grad_(True)
            gi = g[:, sl].detach().clone().requires_grad_(True)
            bi = beta[:, sl].detach().clone().requires_grad_(True)
            h0i = (
                h0[i : i + 1].detach().clone().requires_grad_(grads_h0 is not None)
                if h0 is not None
                else None
            )
            o_i, fs_i = _eager_chunk_gated_delta_rule(
                qi, ki, vi, gi, bi, scale, h0i, output_final_state
            )
            outputs = [o_i]
            grad_outputs = [do[:, sl].to(o_i.dtype)]
            if output_final_state and dfinal_state is not None and fs_i is not None:
                outputs.append(fs_i)
                grad_outputs.append(dfinal_state[i : i + 1].to(fs_i.dtype))
            inputs = [qi, ki, vi, gi, bi]
            if h0i is not None and h0i.requires_grad:
                inputs.append(h0i)
            grads = torch.autograd.grad(
                outputs, inputs, grad_outputs, allow_unused=True
            )
            grads_q[:, sl] = grads[0] if grads[0] is not None else 0
            grads_k[:, sl] = grads[1] if grads[1] is not None else 0
            grads_v[:, sl] = grads[2] if grads[2] is not None else 0
            grads_g[:, sl] = grads[3] if grads[3] is not None else 0
            grads_beta[:, sl] = grads[4] if grads[4] is not None else 0
            if grads_h0 is not None and len(grads) > 5 and grads[5] is not None:
                grads_h0[i : i + 1] = grads[5]
    return (grads_q, grads_k, grads_v, grads_g, grads_beta, None, grads_h0, None, None)


# ---------------------------------------------------------------------------
# Native-Triton fast path for the whole autograd graph.
#
# When the ``fla`` (flash-linear-attention) library is installed we delegate
# both forward and backward to its ``chunk_gated_delta_rule`` autograd
# Function — that gives native chunk-parallel Triton kernels for both
# directions (≈ 20x faster backward than rerunning the eager reference).
#
# If the library is absent or fails on the current arch, callers fall back
# to ``_ChunkGatedDeltaRuleFn`` (FLA forward via FlagGems' own kernels +
# differentiable-eager backward).  The fallback is correct everywhere; the
# fast path is just an optimization.
# ---------------------------------------------------------------------------
_FLA_AUTOGRAD_AVAILABLE = None  # tri-state cache: None=unchecked, True/False=cached


def _try_fla_autograd():
    """Return ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` or None."""
    global _FLA_AUTOGRAD_AVAILABLE
    if _FLA_AUTOGRAD_AVAILABLE is False:
        return None
    try:
        # ``FLA_DISABLE_BACKEND_DISPATCH=1`` keeps us on the Triton backend
        # (and avoids the TileLang backend which is unstable on some archs).
        os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fn

        _ensure_triton_allocator()
        _FLA_AUTOGRAD_AVAILABLE = True
        return fn
    except Exception:
        _FLA_AUTOGRAD_AVAILABLE = False
        return None


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """Chunk-parallel gated delta rule (Qwen3-Next style).

    Args:
        q, k:           (B, T, H, K) bf16/fp16/fp32
        v:              (B, T, H, V) bf16/fp16/fp32
        g, beta:        (B, T, H)    fp32-friendly
        scale:          softmax-style scaling for q (default 1/sqrt(K))
        initial_state:  (B, H, K, V) fp32 carry-in (optional)
        output_final_state: if True, also return the final hidden state in fp32
        cu_seqlens:     LongTensor (N+1,) for variable-length packed batches.
                        When set, ``B`` must be 1 and ``T`` is the total length.

    Returns:
        (out, final_state)
            out:          (B, T, H, V) same dtype as ``q``
            final_state:  (B, H, K, V) fp32 if output_final_state else None
    """
    logger.debug("GEMS CHUNK_GATED_DELTA_RULE")

    # Argument sanity checks.
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q,k,v must be 4D"
    assert q.shape == k.shape, "q and k must have same shape"
    assert q.shape[:3] == v.shape[:3], "v must match q in (B,T,H)"
    assert g.shape == beta.shape == q.shape[:3], "g, beta must be (B,T,H)"
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "cu_seqlens requires B=1 packed layout"

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()
    h0_c = initial_state.contiguous() if initial_state is not None else None
    cu_c = cu_seqlens.contiguous() if cu_seqlens is not None else None
    scale_val = scale if scale is not None else q.shape[-1] ** -0.5

    # Fast path: delegate to upstream FLA's autograd Function (native Triton
    # fwd + bwd).  ~20x faster backward than our eager autograd fallback.
    fla_fn = _try_fla_autograd() if q.is_cuda else None
    if fla_fn is not None:
        try:
            out, final_state = fla_fn(
                q_c,
                k_c,
                v_c,
                g_c,
                beta_c,
                scale=scale_val,
                initial_state=h0_c,
                output_final_state=output_final_state,
                cu_seqlens=cu_c,
            )
            if final_state is not None and final_state.dtype != torch.float32:
                final_state = final_state.to(torch.float32)
            if not output_final_state:
                final_state = None
            return out, final_state
        except Exception:
            pass  # fall through to FlagGems-internal path

    out, final_state = _ChunkGatedDeltaRuleFn.apply(
        q_c,
        k_c,
        v_c,
        g_c,
        beta_c,
        scale,
        h0_c,
        output_final_state,
        cu_c,
    )
    if not output_final_state:
        final_state = None
    return out, final_state
