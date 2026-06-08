import logging

import torch

from flag_gems.runtime.backend._ascend.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd as _ascend_fused_recurrent_gated_delta_rule_fwd,
)

logger = logging.getLogger(__name__)


def _contiguous_if_needed(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS SUNRISE FUSED RECURRENT GATED DELTA RULE FWD")

    # Reuse the simpler contiguous-layout kernel path for Sunrise/PTPU.
    # The generic FLA implementation shows large forward drift on PTPU when
    # `use_qk_l2norm_in_kernel=False`, while this layout-stable variant matches
    # the PyTorch reference. We normalize views here instead of touching the
    # shared/public FLA implementation.
    return _ascend_fused_recurrent_gated_delta_rule_fwd(
        q=_contiguous_if_needed(q),
        k=_contiguous_if_needed(k),
        v=_contiguous_if_needed(v),
        g=_contiguous_if_needed(g),
        beta=_contiguous_if_needed(beta),
        scale=scale,
        initial_state=_contiguous_if_needed(initial_state),
        inplace_final_state=inplace_final_state,
        cu_seqlens=_contiguous_if_needed(cu_seqlens),
        ssm_state_indices=_contiguous_if_needed(ssm_state_indices),
        num_accepted_tokens=_contiguous_if_needed(num_accepted_tokens),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
