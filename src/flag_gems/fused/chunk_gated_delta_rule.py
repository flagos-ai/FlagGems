import logging

import torch
import torch.nn.functional as F

from flag_gems.fused.FLA.chunk import chunk_gated_delta_rule_fwd

logger = logging.getLogger(__name__)


def _maybe_head_first_to_seq_first(x: torch.Tensor, head_first: bool) -> torch.Tensor:
    if head_first:
        return x.transpose(1, 2).contiguous()
    return x.contiguous()


def _maybe_seq_first_to_head_first(x: torch.Tensor, head_first: bool) -> torch.Tensor:
    if head_first:
        return x.transpose(1, 2).contiguous()
    return x


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = True,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
):
    logger.debug("GEMS CHUNK_GATED_DELTA_RULE")

    if BT != 64:
        raise ValueError("chunk_gated_delta_rule currently supports BT=64 only")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k and v must be 4D tensors")
    if beta.dim() != 3 or g.dim() != 3:
        raise ValueError("beta and g must be 3D tensors")

    q = _maybe_head_first_to_seq_first(q, head_first)
    k = _maybe_head_first_to_seq_first(k, head_first)
    v = _maybe_head_first_to_seq_first(v, head_first)
    beta = _maybe_head_first_to_seq_first(beta, head_first)
    g = _maybe_head_first_to_seq_first(g, head_first)

    if q.shape != k.shape:
        raise ValueError(
            f"q and k must have the same shape, but got {q.shape} and {k.shape}"
        )
    if q.shape[:2] != v.shape[:2]:
        raise ValueError("q/k and v must share batch and sequence dimensions")
    if v.shape[:3] != beta.shape or v.shape[:3] != g.shape:
        raise ValueError("beta and g must match v on [B, T, H]")

    B, _, _, K = q.shape
    H = v.shape[2]
    V = v.shape[3]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if initial_state is None:
        initial_state = torch.zeros((N, H, K, V), device=q.device, dtype=q.dtype)
    elif initial_state.shape != (N, H, K, V):
        raise ValueError(
            "initial_state must have shape "
            f"{(N, H, K, V)}, but got {tuple(initial_state.shape)}"
        )

    if scale is None:
        scale = K**-0.5

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)

    _, o, _, final_state, *_ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = _maybe_seq_first_to_head_first(o, head_first)
    return o, final_state
