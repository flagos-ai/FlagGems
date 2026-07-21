# TopK routing — Ascend CANN native operator implementation.
# Called from impl/topk.py (thin dispatch layer).

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.layers.moe.topk import StandardTopKOutput

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


def _l1_norm(x: torch.Tensor) -> torch.Tensor:
    """In-place L1 row-normalization: x /= sum(x, dim=-1)."""
    s = x.sum(dim=-1, keepdim=True)
    s = s.clamp(min=torch.finfo(s.dtype).eps)
    return x.div_(s)


# ═══════════════════════════════════════════════════════════════════════════
# Ascend NPU: CANN native TopK
# ═══════════════════════════════════════════════════════════════════════════


def topk_ascend(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":
    K = topk_config.top_k
    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias

    # ── Fast path: simple softmax top-k (no grouped, no bias) ────────────
    if not use_grouped_topk and correction_bias is None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=K,
        )
        if renormalize:
            _l1_norm(
                topk_weights
                if topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1]
            )
        topk_weights = topk_weights.to(torch.float32)

    # ── sqrtsoftplus (DSV4 noaux_tc): torch path ─────────────────────────
    elif topk_config.scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(router_logits.float()).sqrt()
        scores_for_choice = (
            scores + correction_bias.unsqueeze(0).float()
            if correction_bias is not None
            else scores
        )
        _, topk_ids = torch.topk(scores_for_choice, k=K, dim=-1, sorted=False)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = scores.gather(1, topk_ids)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights = topk_weights * topk_config.routed_scaling_factor
        topk_weights = topk_weights.to(torch.float32)

    # ── Grouped / bias / sigmoid / padded-token path ─────────────────────
    elif (
        correction_bias is not None
        or topk_config.scoring_func == "sigmoid"
        or num_token_non_padded is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=K,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            k_group=topk_config.topk_group if use_grouped_topk else 1,
            group_count=topk_config.num_expert_group if use_grouped_topk else 1,
            group_select_mode=(1 if use_grouped_topk else 0),
            renorm=0,
            norm_type=1,  # 1 = sigmoid, 0 = softmax
            routed_scaling_factor=(
                1 if renormalize else topk_config.routed_scaling_factor
            ),
            eps=float(1e-20),
        )
        topk_weights = topk_weights.to(torch.float32)

    # ── Fallback: torch native ───────────────────────────────────────────
    else:
        from sglang.srt.layers.moe.topk import select_experts

        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    # ── Post-processing (all Ascend paths) ───────────────────────────────
    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    get_global_experts_capturer().capture(layer_id, topk_ids)
    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
