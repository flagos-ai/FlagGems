import torch
import torch.nn.functional as F

from flag_gems import runtime
from flag_gems.fused.FLA import chunk_gated_delta_rule_fwd
from flag_gems.fused.FLA.chunk_gated_delta_direct import (
    can_use_chunk_gated_delta_rule_direct,
    chunk_gated_delta_rule_direct_fwd,
)
from flag_gems.fused.FLA.fused_recurrent import fused_recurrent_gated_delta_rule_fwd


def _as_seq_first(
    x: torch.Tensor,
    *,
    name: str,
    head_first: bool,
    expected_ndim: int,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != expected_ndim:
        raise ValueError(f"{name} must be {expected_ndim}D, got shape {tuple(x.shape)}")
    if head_first:
        return x.transpose(1, 2)
    return x


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    B, T, Hg, K = q.shape
    Bk, Tk, Hk, Kk = k.shape
    Bv, Tv, H, V = v.shape

    tensors = {"k": k, "v": v, "beta": beta, "g": g}
    for name, tensor in tensors.items():
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if tensor.dtype != q.dtype:
            raise ValueError(f"{name} must have the same dtype as q")

    if (Bk, Tk, Hk, Kk) != (B, T, Hg, K):
        raise ValueError(
            "q and k must have matching [B, T, Hq, K] shapes after layout conversion"
        )
    if (Bv, Tv) != (B, T):
        raise ValueError("v must have matching B and T dimensions with q/k")
    if H % Hg != 0:
        raise ValueError("the q/k head count must divide the v head count")
    if beta.shape != (B, T, H):
        raise ValueError(
            f"beta must have shape {(B, T, H)} after layout conversion, got {tuple(beta.shape)}"
        )
    if g.shape != (B, T, H):
        raise ValueError(
            f"g must have shape {(B, T, H)} after layout conversion, got {tuple(g.shape)}"
        )
    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D tensor")
        if cu_seqlens.dtype != torch.long:
            raise ValueError("cu_seqlens must have dtype torch.long")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q")
        if B != 1:
            raise ValueError("cu_seqlens packed varlen inputs must use B=1")

    if initial_state is not None:
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        if initial_state.dtype != q.dtype:
            raise ValueError("initial_state must have the same dtype as q")
        expected_n = B if cu_seqlens is None else cu_seqlens.numel() - 1
        expected_shape = (expected_n, H, K, V)
        if initial_state.shape != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, got {tuple(initial_state.shape)}"
            )


def _direct_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _is_iluvatar_backend() -> bool:
    return getattr(runtime.device, "vendor_name", "").lower() == "iluvatar"


def _torch_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    beta_float = beta.float()
    g_float = g.float()
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    heads_per_group = H // Hg
    out = torch.empty_like(v_float)
    final_state = None
    if output_final_state:
        n_state = B if cu_seqlens is None else cu_seqlens.numel() - 1
        final_state = torch.empty(
            n_state, H, K, V, device=v.device, dtype=torch.float32
        )

    if cu_seqlens is None:
        spans = [(batch, batch, 0, T) for batch in range(B)]
    else:
        cu_cpu = cu_seqlens.detach().cpu().tolist()
        spans = [(0, seq, cu_cpu[seq], cu_cpu[seq + 1]) for seq in range(len(cu_cpu) - 1)]

    for batch_idx, state_idx, start, end in spans:
        if initial_state is None:
            h = torch.zeros(H, K, V, device=v.device, dtype=torch.float32)
        else:
            h = initial_state[state_idx].float().clone()
        for t in range(start, end):
            q_t = q_float[batch_idx, t]
            k_t = k_float[batch_idx, t]
            for h_idx in range(H):
                qk_head = h_idx // heads_per_group
                h[h_idx] *= torch.exp(g_float[batch_idx, t, h_idx])
                k_vec = k_t[qk_head]
                residual = v_float[batch_idx, t, h_idx] - torch.matmul(k_vec, h[h_idx])
                update = residual * beta_float[batch_idx, t, h_idx]
                h[h_idx] += k_vec[:, None] * update[None, :]
                out[batch_idx, t, h_idx] = torch.matmul(q_t[qk_head] * scale, h[h_idx])
        if final_state is not None:
            final_state[state_idx] = h

    return out.to(v.dtype), final_state


def _recurrent_kernel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, _, K = q.shape
    H, V = v.shape[2], v.shape[3]
    if cu_seqlens is None:
        q_recurrent = q.contiguous().reshape(1, B * T, q.shape[2], K)
        k_recurrent = k.contiguous().reshape(1, B * T, k.shape[2], K)
        v_recurrent = v.contiguous().reshape(1, B * T, H, V)
        beta_recurrent = beta.contiguous().reshape(1, B * T, H)
        g_recurrent = g.contiguous().reshape(1, B * T, H)
        cu_recurrent = torch.arange(
            0, (B + 1) * T, T, device=q.device, dtype=torch.long
        )
        ssm_state_indices = (
            torch.arange(B, device=q.device, dtype=torch.long)
            .view(B, 1)
            .expand(B, T)
            .contiguous()
        )
        state_count = B
    else:
        q_recurrent = q.contiguous()
        k_recurrent = k.contiguous()
        v_recurrent = v.contiguous()
        beta_recurrent = beta.contiguous()
        g_recurrent = g.contiguous()
        cu_recurrent = cu_seqlens
        ssm_state_indices = None
        state_count = cu_seqlens.numel() - 1

    if initial_state is None:
        recurrent_state = q.new_zeros(state_count, H, K, V)
    else:
        recurrent_state = initial_state.contiguous().clone()

    if cu_seqlens is not None:
        return _torch_recurrent_fwd(
            q=q_recurrent,
            k=k_recurrent,
            v=v_recurrent,
            beta=beta_recurrent,
            g=g_recurrent,
            scale=scale,
            initial_state=recurrent_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_recurrent,
        )

    out, final_state = fused_recurrent_gated_delta_rule_fwd(
        q=q_recurrent,
        k=k_recurrent,
        v=v_recurrent,
        g=g_recurrent,
        beta=beta_recurrent,
        scale=float(scale),
        initial_state=recurrent_state,
        inplace_final_state=output_final_state,
        cu_seqlens=cu_recurrent,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    )
    out = out.reshape(B, T, H, V)
    return out, final_state if output_final_state else None


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = True,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Public wrapper for the chunk gated delta rule forward operator.

    Inputs follow common FLA layouts:
    - ``head_first=True``: q/k/v are ``[B, H, T, D]`` and beta/g are ``[B, H, T]``.
    - ``head_first=False``: q/k/v are ``[B, T, H, D]`` and beta/g are ``[B, T, H]``.

    q/k may use fewer heads than v when the q/k head count divides the v head count.
    """
    if BT != 64:
        raise ValueError("chunk_gated_delta_rule currently supports only BT=64")

    q_seq = _as_seq_first(q, name="q", head_first=head_first, expected_ndim=4)
    k_seq = _as_seq_first(k, name="k", head_first=head_first, expected_ndim=4)
    v_seq = _as_seq_first(v, name="v", head_first=head_first, expected_ndim=4)
    beta_seq = _as_seq_first(beta, name="beta", head_first=head_first, expected_ndim=3)
    g_seq = _as_seq_first(g, name="g", head_first=head_first, expected_ndim=3)

    _validate_inputs(q_seq, k_seq, v_seq, beta_seq, g_seq, initial_state, cu_seqlens)

    if scale is None:
        scale = k_seq.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q_seq = F.normalize(q_seq, p=2.0, dim=-1, eps=1e-6)
        k_seq = F.normalize(k_seq, p=2.0, dim=-1, eps=1e-6)

    B, T, Hg, K = q_seq.shape
    H, V = v_seq.shape[2], v_seq.shape[3]
    if (
        initial_state is None
        and cu_seqlens is None
        and T <= 128
        and K <= 128
        and V <= 128
        and H % Hg == 0
    ):
        q_direct = _direct_contiguous(q_seq)
        k_direct = _direct_contiguous(k_seq)
        v_direct = _direct_contiguous(v_seq)
        g_direct = _direct_contiguous(g_seq)
        beta_direct = _direct_contiguous(beta_seq)
        if can_use_chunk_gated_delta_rule_direct(
            q=q_direct,
            k=k_direct,
            v=v_direct,
            g=g_direct,
            beta=beta_direct,
            initial_state=None,
            cu_seqlens=None,
        ):
            o, final_state = chunk_gated_delta_rule_direct_fwd(
                q=q_direct,
                k=k_direct,
                v=v_direct,
                g=g_direct,
                beta=beta_direct,
                scale=float(scale),
                initial_state=None,
                output_final_state=output_final_state,
            )
            if head_first:
                o = o.transpose(1, 2)
            return o, final_state

    if _is_iluvatar_backend():
        o, final_state = _recurrent_kernel_fwd(
            q=q_seq,
            k=k_seq,
            v=v_seq,
            beta=beta_seq,
            g=g_seq,
            scale=float(scale),
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        if head_first:
            o = o.transpose(1, 2)
        return o, final_state

    _, o, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q_seq,
        k=k_seq,
        v=v_seq,
        g=g_seq,
        beta=beta_seq,
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    if head_first:
        o = o.transpose(1, 2)
    return o, final_state
