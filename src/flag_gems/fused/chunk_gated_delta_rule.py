import logging

import torch
import torch.nn.functional as F

from flag_gems.fused.FLA.chunk import chunk_gated_delta_rule_fwd

logger = logging.getLogger(__name__)


def _pad_to_multiple(x: torch.Tensor, pad_len: int):
    if pad_len == 0:
        return x
    if x.dim() == 4:
        return F.pad(x, (0, 0, 0, pad_len))
    if x.dim() == 3:
        return F.pad(x.unsqueeze(-1), (0, 0, 0, pad_len)).squeeze(-1)
    raise ValueError(f"Unsupported input rank: {x.dim()}")


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
):
    """
    Fused chunked gated delta rule (Qwen3-Next style).

    Expected input format when `head_first=True`:
        q, k, v: [B, H, T, D]
        beta, g: [B, H, T]
    """
    logger.debug("GEMS CHUNK_GATED_DELTA_RULE")
    if BT != 64:
        raise ValueError("chunk_gated_delta_rule currently supports BT=64 only")
    if q.dim() != 4:
        raise ValueError(f"Expected 4D q/k/v, but got {q.dim()}D")
    if beta.dim() != 3 or g.dim() != 3:
        raise ValueError("beta and g must be 3D tensors with shape [B, H, T]")

    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)
        g = g.transpose(1, 2)

    seq_len = q.shape[2]
    pad_len = (BT - seq_len % BT) % BT
    if pad_len:
        q = _pad_to_multiple(q, pad_len)
        k = _pad_to_multiple(k, pad_len)
        v = _pad_to_multiple(v, pad_len)
        beta = _pad_to_multiple(beta, pad_len)
        g = _pad_to_multiple(g, pad_len)

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()

    scale = q.shape[-1] ** -0.5
    _, o, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
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

    o = o.transpose(1, 2)
    if pad_len:
        o = o[:, :, :seq_len, :]
    if not head_first:
        o = o.transpose(1, 2)
    return o, final_state
