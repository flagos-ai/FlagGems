"""
chunk_gated_delta.py – Chunk Gated Delta Rule for FlagGems
==========================================================

Forward recurrence (per head, sequential over T):
    r_t = v_t  −  k_t @ S_{t-1}          # residual
    S_t = S_{t-1}  +  β_t · outer(k_t, r_t)   # rank-1 state update
    o_t = q_t @ S_t                        # output

Performance design
------------------
* Column-partition parallelism: each Triton program handles one
  (batch, head, col_chunk) triple and stores only a [BD, BDC] slice of
  S in registers.  This reduces register pressure by BD/BDC× and
  increases concurrent programs by the same factor, dramatically
  improving GPU occupancy.
* Autotuning over BDC (column-block width), num_warps, and num_stages
  (software-pipeline depth) via ``triton.autotune``.
* No ``tl.make_block_ptr`` – plain scalar-offset loads avoid the LLVM
  struct size-mismatch error.
* Backward is a pure-PyTorch analytic reverse scan – correct,
  differentiable, and easy to audit.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

__all__ = [
    "torch_chunk_gated_delta_rule",
    "chunk_gated_delta_rule",
    "FlagOS_ChunkGatedDelta",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PyTorch Reference  (used for validation and gradient checking)
# ─────────────────────────────────────────────────────────────────────────────


def torch_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Sequential PyTorch reference implementation.

    Args:
        q, k, v : (B, H, T, D)
        beta    : (B, H, T)

    Returns:
        o : (B, H, T, D)
    """
    B, H, T, D = q.shape
    o = torch.zeros_like(q)
    for b in range(B):
        for h in range(H):
            S = q.new_zeros(D, D)
            for t in range(T):
                k_t = k[b, h, t]  # [D]
                r_t = v[b, h, t] - k_t @ S  # [D]
                S = S + beta[b, h, t] * torch.outer(k_t, r_t)
                o[b, h, t] = q[b, h, t] @ S
    return o


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Triton Forward Kernel – Column-Partitioned with Autotuning
# ─────────────────────────────────────────────────────────────────────────────


def _prune_cgd_configs(configs, named_args, **kwargs):
    """
    Drop autotune configs where BDC > BD (column block wider than the
    padded head dim D).  Such configs waste registers by keeping unused
    padding columns in S_local.  Falls back to a single-block config
    covering all BD columns for very small D.
    """
    # Use D (a runtime int arg) to compute BD; fall back to 64 if absent.
    D = named_args.get("D", 64)
    BD = triton.next_power_of_2(D)
    pruned = [c for c in configs if c.kwargs.get("BDC", 16) <= BD]
    if not pruned:
        # For very small D (e.g. D=8), use 1 block covering all BD columns.
        return [triton.Config({"BDC": BD}, num_warps=4, num_stages=1)]
    return pruned


@triton.autotune(
    configs=[
        # BDC=8 – for very small D (D≤16) or extra parallelism at large D
        triton.Config({"BDC": 8}, num_warps=4, num_stages=1),
        triton.Config({"BDC": 8}, num_warps=8, num_stages=1),
        # BDC=16 – smallest tested column block; 4 programs per (b,h) for D=64
        triton.Config({"BDC": 16}, num_warps=4, num_stages=1),
        triton.Config({"BDC": 16}, num_warps=4, num_stages=2),
        triton.Config({"BDC": 16}, num_warps=8, num_stages=1),
        triton.Config({"BDC": 16}, num_warps=8, num_stages=2),
        # BDC=32 – balanced; 2 programs per (b,h) for D=64
        triton.Config({"BDC": 32}, num_warps=4, num_stages=1),
        triton.Config({"BDC": 32}, num_warps=4, num_stages=2),
        triton.Config({"BDC": 32}, num_warps=8, num_stages=1),
        triton.Config({"BDC": 32}, num_warps=8, num_stages=2),
        # BDC=64 – largest tested block; 1 program per (b,h) for D=64
        triton.Config({"BDC": 64}, num_warps=4, num_stages=1),
        triton.Config({"BDC": 64}, num_warps=8, num_stages=1),
    ],
    key=["T", "BD"],
    prune_configs_by={"early_config_prune": _prune_cgd_configs},
)
@triton.jit
def _cgd_fwd_kernel(
    # ── tensor pointers ──────────────────────────────────────────────────────
    Q,
    K,
    V,
    Beta,
    O,
    # ── strides for Q  (B, H, T, D) ─────────────────────────────────────────
    sq_b,
    sq_h,
    sq_t,
    sq_d,
    # ── strides for K ────────────────────────────────────────────────────────
    sk_b,
    sk_h,
    sk_t,
    sk_d,
    # ── strides for V ────────────────────────────────────────────────────────
    sv_b,
    sv_h,
    sv_t,
    sv_d,
    # ── strides for Beta  (B, H, T) ──────────────────────────────────────────
    sb_b,
    sb_h,
    sb_t,
    # ── strides for O ────────────────────────────────────────────────────────
    so_b,
    so_h,
    so_t,
    so_d,
    # ── runtime scalars ──────────────────────────────────────────────────────
    H,  # int – number of heads
    T,  # int – sequence length
    D,  # int – head dimension
    # ── compile-time constants ────────────────────────────────────────────────
    BD: tl.constexpr,  # next_power_of_2(D) – full row block (padded D)
    BDC: tl.constexpr,  # column block width; BD // BDC programs per (b,h)
):
    """
    Column-partitioned forward kernel.

    Grid: ``(B*H,  BD // BDC)`` programs.

    Each program handles **all T timesteps** for one (batch, head, col_chunk)
    triple, maintaining only the column slice
        S_local  [BD, BDC]  =  S[:, col_start : col_start + BDC]
    in registers.  All delta-rule operations decompose cleanly over columns:

      kS_col[j] = Σ_i  k_t[i] · S_local[i, j]   (needs full k_t row)
      r_t_col   = v_t_col − kS_col               (local)
      S_local  += β_t · outer(k_t, r_t_col)      (local)
      o_t_col[j]= Σ_i  q_t[i] · S_local[i, j]   (needs full q_t row)

    No cross-program synchronisation is required.
    """
    pid_bh = tl.program_id(0)  # (batch, head) index
    pid_col = tl.program_id(1)  # column-block index

    b = pid_bh // H
    h = pid_bh % H
    col_start = pid_col * BDC

    # Index arrays
    d_row = tl.arange(0, BD)  # [BD]  – full row dim
    d_col = col_start + tl.arange(0, BDC)  # [BDC] – this column slice

    row_mask = d_row < D  # [BD]
    col_mask = d_col < D  # [BDC]

    # Base pointers (time-independent offsets to the (b, h) slice)
    q_base = Q + b * sq_b + h * sq_h
    k_base = K + b * sk_b + h * sk_h
    v_base = V + b * sv_b + h * sv_h
    beta_base = Beta + b * sb_b + h * sb_h
    o_base = O + b * so_b + h * so_h

    # Column slice of the state matrix S  [BD, BDC], initialised to 0
    S_local = tl.zeros([BD, BDC], dtype=tl.float32)

    for t in range(T):
        # ── load this timestep ───────────────────────────────────────────────
        k_t = tl.load(k_base + t * sk_t + d_row * sk_d, mask=row_mask, other=0.0).to(
            tl.float32
        )  # [BD]
        v_t = tl.load(v_base + t * sv_t + d_col * sv_d, mask=col_mask, other=0.0).to(
            tl.float32
        )  # [BDC]
        bt = tl.load(beta_base + t * sb_t).to(tl.float32)  # scalar

        # ── delta-rule state update ──────────────────────────────────────────
        # kS_col[j] = Σ_i  k_t[i] · S_local[i, j]
        kS = tl.sum(k_t[:, None] * S_local, axis=0)  # [BDC]

        # r_t_col = v_t_col − kS_col
        r_t = v_t - kS  # [BDC]

        # S_local += β_t · outer(k_t, r_t_col)   [BD, BDC]
        S_local = S_local + bt * (k_t[:, None] * r_t[None, :])

        # ── output ──────────────────────────────────────────────────────────
        q_t = tl.load(q_base + t * sq_t + d_row * sq_d, mask=row_mask, other=0.0).to(
            tl.float32
        )  # [BD]

        # o_t_col[j] = Σ_i  q_t[i] · S_local[i, j]
        o_t = tl.sum(q_t[:, None] * S_local, axis=0)  # [BDC]

        tl.store(o_base + t * so_t + d_col * so_d, o_t, mask=col_mask)


def _triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    BT: int,  # kept for API compatibility; actual chunk size managed by BDC
) -> torch.Tensor:
    """Host-side launcher for ``_cgd_fwd_kernel``."""
    B, H, T, D = q.shape
    BD = triton.next_power_of_2(D)

    # Ensure inputs are contiguous so strides are predictable
    q, k, v, beta = (t.contiguous() for t in (q, k, v, beta))

    o = torch.empty_like(q)

    # Grid: dim-0 = (batch, head) pairs; dim-1 = column blocks.
    # BDC is selected by autotuning; capped at BD via ``_prune_cgd_configs``.
    # ``triton.cdiv(BD, BDC)`` gives the number of column-block programs.
    grid = lambda meta: (B * H, triton.cdiv(BD, meta["BDC"]))

    _cgd_fwd_kernel[grid](
        q,
        k,
        v,
        beta,
        o,
        # strides Q
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # strides K
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        # strides V
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        # strides Beta
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        # strides O
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        # runtime dims
        H=H,
        T=T,
        D=D,
        # compile-time block dims (BDC chosen by autotune)
        BD=BD,
    )
    return o


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyTorch Analytic Backward  (reverse sequential scan)
# ─────────────────────────────────────────────────────────────────────────────


def _torch_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ∂L/∂q, ∂L/∂k, ∂L/∂v, ∂L/∂β via a two-pass algorithm:

    Pass 1 (forward)  – re-run the recurrence to cache  S_{t-1}  and  r_t.
    Pass 2 (backward) – propagate  dS  backward, accumulating gradients.

    Derivation (index notation, L = scalar loss)
    --------------------------------------------
    From  o_t = q_t @ S_t:
        dq_t  = S_t  @ do_t
        dS_t += outer(q_t, do_t)

    From  S_t = S_{t-1} + β_t · outer(k_t, r_t):
        dβ_t  = k_t  @ (dS_t @ r_t)
        dk_t += β_t  *  dS_t @ r_t
        dr_t  = β_t  *  dS_t.T @ k_t
        dS_{t-1} = dS_t  −  outer(k_t, dr_t)   ← pass-through minus kS term

    From  r_t = v_t − k_t @ S_{t-1}:
        dv_t += dr_t
        dk_t -= S_{t-1} @ dr_t
        (dS_{t-1} contribution already folded in above)
    """
    B, H, T, D = q.shape

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)

    for b in range(B):
        for h in range(H):
            # ── Pass 1: forward to cache S_{t-1} and r_t ──────────────────
            S_prevs = []  # S_{t-1}  for each t
            residuals = []  # r_t      for each t
            S = q.new_zeros(D, D)

            for t in range(T):
                S_prevs.append(S.clone())
                k_t = k[b, h, t]
                r_t = v[b, h, t] - k_t @ S
                residuals.append(r_t.clone())
                S = S + beta[b, h, t] * torch.outer(k_t, r_t)

            # ── Pass 2: backward scan ──────────────────────────────────────
            dS = q.new_zeros(D, D)  # ∂L/∂S_t; starts at 0 (= ∂L/∂S_T)

            for t in range(T - 1, -1, -1):
                S_prev = S_prevs[t]
                k_t = k[b, h, t]
                r_t = residuals[t]
                bt = beta[b, h, t]
                do_t = do[b, h, t]

                # Reconstruct S_t for the dq update
                S_t = S_prev + bt * torch.outer(k_t, r_t)

                # ∂L/∂q_t  and  accumulate ∂L/∂S_t from o_t
                dq[b, h, t] = S_t @ do_t
                dS = dS + torch.outer(q[b, h, t], do_t)

                # Gradients through S_t = S_prev + β · outer(k_t, r_t)
                dSr = dS @ r_t  # [D]
                dbeta[b, h, t] = torch.dot(k_t, dSr)
                dk[b, h, t] += bt * dSr
                dr_t = bt * (dS.T @ k_t)  # [D]

                # Gradients through r_t = v_t − k_t @ S_prev
                dv[b, h, t] += dr_t
                dk[b, h, t] -= S_prev @ dr_t

                # Pass ∂L/∂S_{t-1} to the next (earlier) step
                dS = dS - torch.outer(k_t, dr_t)

    return dq, dk, dv, dbeta


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Custom Autograd Function
# ─────────────────────────────────────────────────────────────────────────────


class _ChunkGatedDeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, BT):
        o = _triton_forward(q, k, v, beta, BT)
        ctx.save_for_backward(q, k, v, beta)
        ctx.BT = BT
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, beta = ctx.saved_tensors
        dq, dk, dv, dbeta = _torch_backward(q, k, v, beta, do.contiguous())
        return dq, dk, dv, dbeta, None  # no grad for BT


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public functional API
# ─────────────────────────────────────────────────────────────────────────────


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    BT: int = 16,
) -> torch.Tensor:
    """
    Triton-accelerated chunk gated delta rule (forward + analytic backward).

    Args:
        q, k, v : (B, H, T, D)  – query / key / value
        beta    : (B, H, T)     – per-step gate / learning rate
        BT      : chunk size    – kept for API compatibility

    Returns:
        o : (B, H, T, D)
    """
    return _ChunkGatedDeltaFn.apply(q, k, v, beta, BT)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  nn.Module wrapper
# ─────────────────────────────────────────────────────────────────────────────


class FlagOS_ChunkGatedDelta(nn.Module):
    """
    ``nn.Module`` wrapper around ``chunk_gated_delta_rule``.

    Usage::
        layer = FlagOS_ChunkGatedDelta(BT=16)
        o = layer(q, k, v, beta)
    """

    def __init__(self, BT: int = 16) -> None:
        super().__init__()
        self.BT = BT

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return chunk_gated_delta_rule(q, k, v, beta, BT=self.BT)
