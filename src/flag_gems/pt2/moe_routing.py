# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Transparent PT2 adapters for the FlagGems MoE routing kernels.

The adapters in this module only replace Python launch control.  Eager mode
continues to call the public FlagGems launchers, while compiled mode launches
the exact same common/NVIDIA Triton ``JITFunction`` objects through
``torch.library.triton_op`` and ``torch.library.wrap_triton``.

``grouped_topk`` remains the original two-stage algorithm (group scoring,
then expert selection).  ``topk_softplus_sqrt`` remains two explicitly
different structural families: direct top-k and hash-table routing.  Token
count is the runtime grid dimension; expert/group/top-k configuration,
branch identity, dtype, and layout are compile specializations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.fused.grouped_topk import (
    group_idx_and_topk_triton as GROUPED_TOPK_SELECT_JIT,
)
from flag_gems.fused.grouped_topk import grouped_topk as _eager_grouped_topk
from flag_gems.fused.grouped_topk import topk_with_k2_triton as GROUPED_TOPK_GROUP_JIT
from flag_gems.fused.topk_softplus_sqrt import (
    _fused_topk_kernel as TOPK_SOFTPLUS_SQRT_JIT,
)
from flag_gems.fused.topk_softplus_sqrt import (
    _hash_kernel as TOPK_HASH_SOFTPLUS_SQRT_JIT,
)
from flag_gems.fused.topk_softplus_sqrt import (
    topk_softplus_sqrt as _eager_topk_softplus_sqrt,
)
from flag_gems.pt2.manifest import CompileKind, CompileOpSpec, register_compile_spec

_HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(
    torch.library, "wrap_triton"
)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TOPK_INDEX_DTYPES = (torch.int32, torch.uint32, torch.int64)
_HASH_DTYPES = (torch.int32, torch.int64)


def supports_pt2_moe_routing() -> bool:
    """Return whether this Torch build exposes transparent Triton APIs."""

    return _HAS_TRITON_OP


def uses_common_moe_routing_kernels(
    grouped_topk_impl: object,
    topk_softplus_sqrt_impl: object,
) -> bool:
    """Check that a public/vendor export still names the captured kernels.

    A vendor or optional C++ installation may replace a public FlagGems
    launcher.  In that case the plugin must retain that launcher instead of
    silently compiling the common/NVIDIA adapter around a different kernel.
    """

    return (
        grouped_topk_impl is _eager_grouped_topk
        and topk_softplus_sqrt_impl is _eager_topk_softplus_sqrt
    )


def _check_same_device(reference: torch.Tensor, other: torch.Tensor) -> None:
    torch._check(other.device == reference.device)


def _input_triton_dtype(dtype: torch.dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    raise TypeError("routing input must be float16/bfloat16/float32")


def _normalize_grouped_bias(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Preserve the original public launcher's cast/flatten semantics.  These
    # are ordinary ATen operations and remain visible to Dynamo/Inductor.
    if bias.dtype != scores.dtype:
        bias = bias.to(scores.dtype)
    if bias.ndim != 1:
        bias = bias.flatten()
    return bias


def _check_grouped_topk_contract(
    scores: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    scoring_func: int,
) -> tuple[int, int]:
    torch._check(scores.ndim == 2)
    num_tokens, num_experts = scores.shape
    torch._check(num_experts > 0)
    torch._check(n_group > 0)
    torch._check(n_group <= 32)
    # The first original kernel computes the top two experts in each group.
    # A one-expert group cannot satisfy that ABI (the plugin's model path
    # already rejects this family); fail closed for direct PT2 callers too.
    torch._check(num_experts > n_group)
    torch._check(num_experts % n_group == 0)
    torch._check(topk_group > 0)
    torch._check(topk_group <= n_group)
    torch._check(topk > 0)
    torch._check(topk <= 32)
    torch._check(topk <= topk_group * (num_experts // n_group))
    torch._check(scoring_func >= 0)
    torch._check(scoring_func <= 1)
    torch._check(scores.dtype in _FLOAT_DTYPES)
    torch._check(scores.stride(1) == 1)

    torch._check(bias.ndim == 1)
    torch._check(bias.shape[0] == num_experts)
    torch._check(bias.dtype == scores.dtype)
    torch._check(bias.stride(0) == 1)
    _check_same_device(scores, bias)
    return num_tokens, num_experts


def _check_topk_softplus_contract(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    has_bias: bool,
) -> tuple[int, int, int]:
    torch._check(gating_output.ndim == 2)
    torch._check(topk_weights.ndim == 2)
    torch._check(topk_indices.ndim == 2)
    torch._check(token_expert_indices.ndim == 2)
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]
    torch._check(num_experts > 0)
    torch._check(topk > 0)
    torch._check(topk <= num_experts)
    torch._check(topk_weights.shape[0] == num_tokens)
    torch._check(topk_indices.shape[0] == num_tokens)
    torch._check(topk_indices.shape[1] == topk)
    torch._check(token_expert_indices.shape[0] == num_tokens)
    torch._check(token_expert_indices.shape[1] == topk)

    torch._check(gating_output.dtype in _FLOAT_DTYPES)
    torch._check(topk_weights.dtype == torch.float32)
    torch._check(topk_indices.dtype in _TOPK_INDEX_DTYPES)
    torch._check(token_expert_indices.dtype == torch.int32)
    _check_same_device(gating_output, topk_weights)
    _check_same_device(gating_output, topk_indices)
    _check_same_device(gating_output, token_expert_indices)

    # Both original kernels compute dense row-major addresses and accept no
    # stride arguments.  Reject layouts their ABI cannot represent.
    torch._check(gating_output.is_contiguous())
    torch._check(topk_weights.is_contiguous())
    torch._check(topk_indices.is_contiguous())
    torch._check(token_expert_indices.is_contiguous())

    if has_bias:
        torch._check(correction_bias.ndim == 1)
        torch._check(correction_bias.shape[0] == num_experts)
        torch._check(correction_bias.dtype in _FLOAT_DTYPES)
        torch._check(correction_bias.is_contiguous())
        _check_same_device(gating_output, correction_bias)
    return num_tokens, num_experts, topk


def _check_hash_contract(
    gating_output: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_tokens: int,
    topk: int,
) -> None:
    torch._check(input_ids.ndim == 1)
    torch._check(input_ids.shape[0] == num_tokens)
    torch._check(input_ids.dtype in _HASH_DTYPES)
    torch._check(input_ids.is_contiguous())
    torch._check(tid2eid.ndim == 2)
    torch._check(tid2eid.shape[1] == topk)
    torch._check(tid2eid.dtype in _HASH_DTYPES)
    torch._check(tid2eid.is_contiguous())
    _check_same_device(gating_output, input_ids)
    _check_same_device(gating_output, tid2eid)


if _HAS_TRITON_OP:

    @torch.library.triton_op(
        "flag_gems_pt2::grouped_topk",
        mutates_args={},
    )
    def _grouped_topk_op(
        scores: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
        bias: torch.Tensor,
        scoring_func: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias = _normalize_grouped_bias(scores, bias)
        num_tokens, num_experts = _check_grouped_topk_contract(
            scores,
            bias,
            n_group,
            topk_group,
            topk,
            scoring_func,
        )

        if scoring_func == 1:
            scores_processed = torch.sigmoid(scores.float()).to(scores.dtype)
        else:
            scores_processed = scores

        group_scores = torch.empty(
            (num_tokens, n_group),
            device=scores.device,
            dtype=scores.dtype,
        )
        topk_values = torch.empty(
            (num_tokens, topk),
            device=scores.device,
            dtype=torch.float32,
        )
        topk_indices = torch.empty(
            (num_tokens, topk),
            device=scores.device,
            dtype=torch.int32,
        )
        if num_tokens == 0:
            return topk_values, topk_indices

        num_experts_per_group = num_experts // n_group
        input_dtype = _input_triton_dtype(scores.dtype)
        block_group_scores = triton.next_power_of_2(num_experts_per_group)
        torch.library.wrap_triton(GROUPED_TOPK_GROUP_JIT)[(num_tokens * n_group,)](
            scores_processed,
            bias,
            group_scores,
            num_experts_per_group,
            n_group,
            scores_processed.stride(0),
            group_scores.stride(0),
            BLOCK_SIZE=block_group_scores,
            INPUT_DTYPE=input_dtype,
        )

        block_group = triton.next_power_of_2(n_group)
        block_expert = triton.next_power_of_2(num_experts)
        torch.library.wrap_triton(GROUPED_TOPK_SELECT_JIT)[(num_tokens,)](
            scores_processed,
            group_scores,
            topk_values,
            topk_indices,
            bias,
            num_tokens,
            n_group,
            topk_group,
            topk,
            num_experts,
            num_experts_per_group,
            routed_scaling_factor,
            scores_processed.stride(0),
            group_scores.stride(0),
            topk_values.stride(0),
            N_GROUP=n_group,
            TOPK_GROUP=topk_group,
            TOPK=topk,
            BLOCK_GROUP=block_group,
            BLOCK_EXPERT=block_expert,
            INPUT_DTYPE=input_dtype,
            renormalize=int(renormalize),
        )
        return topk_values, topk_indices

    @torch.library.triton_op(
        "flag_gems_pt2::topk_softplus_sqrt",
        mutates_args={
            "topk_weights",
            "topk_indices",
            "token_expert_indices",
        },
    )
    def _topk_softplus_sqrt_op(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
        routed_scaling_factor: float,
        correction_bias: torch.Tensor,
        has_bias: bool,
    ) -> None:
        num_tokens, num_experts, topk = _check_topk_softplus_contract(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            correction_bias,
            has_bias,
        )
        if num_tokens == 0:
            return

        torch.library.wrap_triton(TOPK_SOFTPLUS_SQRT_JIT)[(num_tokens,)](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=has_bias,
            BLOCK_E=triton.next_power_of_2(num_experts),
            num_warps=1,
            num_stages=1,
        )

    @torch.library.triton_op(
        "flag_gems_pt2::topk_hash_softplus_sqrt",
        mutates_args={
            "topk_weights",
            "topk_indices",
            "token_expert_indices",
        },
    )
    def _topk_hash_softplus_sqrt_op(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
        routed_scaling_factor: float,
        correction_bias: torch.Tensor,
        has_bias: bool,
        input_ids: torch.Tensor,
        tid2eid: torch.Tensor,
    ) -> None:
        num_tokens, num_experts, topk = _check_topk_softplus_contract(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            correction_bias,
            has_bias,
        )
        _check_hash_contract(
            gating_output,
            input_ids,
            tid2eid,
            num_tokens,
            topk,
        )
        if num_tokens == 0:
            return

        torch.library.wrap_triton(TOPK_HASH_SOFTPLUS_SQRT_JIT)[(num_tokens,)](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias,
            input_ids,
            tid2eid,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=has_bias,
            BLOCK_E=triton.next_power_of_2(num_experts),
            BLOCK_K=triton.next_power_of_2(topk),
            num_warps=1,
            num_stages=1,
        )

else:
    _grouped_topk_op = None
    _topk_softplus_sqrt_op = None
    _topk_hash_softplus_sqrt_op = None


_TRANSPARENT_REQUIRES = (
    "torch.library.triton_op",
    "torch.library.wrap_triton",
    "NVIDIA/common FlagGems kernel identity",
)

GROUPED_TOPK_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::grouped_topk",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.grouped_topk.topk_with_k2_triton + "
            "flag_gems.fused.grouped_topk.group_idx_and_topk_triton"
        ),
        mutates_args=(),
        dynamic_dims=("num_tokens",),
        requires=_TRANSPARENT_REQUIRES,
    )
)

TOPK_SOFTPLUS_SQRT_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::topk_softplus_sqrt",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel="flag_gems.fused.topk_softplus_sqrt._fused_topk_kernel",
        mutates_args=(
            "topk_weights",
            "topk_indices",
            "token_expert_indices",
        ),
        dynamic_dims=("num_tokens",),
        requires=_TRANSPARENT_REQUIRES,
    )
)

TOPK_HASH_SOFTPLUS_SQRT_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::topk_hash_softplus_sqrt",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel="flag_gems.fused.topk_softplus_sqrt._hash_kernel",
        mutates_args=(
            "topk_weights",
            "topk_indices",
            "token_expert_indices",
        ),
        dynamic_dims=("num_tokens",),
        requires=_TRANSPARENT_REQUIRES,
    )
)


def _missing_triton_op() -> RuntimeError:
    return RuntimeError(
        "This Torch build lacks triton_op/wrap_triton; the transparent "
        "FlagGems MoE routing PT2 contracts are unavailable"
    )


def grouped_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the original two-stage grouped-top-k implementation."""

    if torch.compiler.is_compiling():
        if _grouped_topk_op is None:
            raise _missing_triton_op()
        return _grouped_topk_op(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )

    # Define the empty-token contract before constructing Triton's zero grid.
    # Non-empty eager calls retain the unmodified public FlagGems launcher.
    if scores.ndim == 2 and scores.shape[0] == 0:
        normalized_bias = _normalize_grouped_bias(scores, bias)
        _check_grouped_topk_contract(
            scores,
            normalized_bias,
            n_group,
            topk_group,
            topk,
            scoring_func,
        )
        return (
            torch.empty((0, topk), dtype=torch.float32, device=scores.device),
            torch.empty((0, topk), dtype=torch.int32, device=scores.device),
        )
    return _eager_grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
) -> None:
    """Run the original direct or hash softplus-sqrt routing kernel."""

    if (input_ids is None) != (tid2eid is None):
        raise ValueError("input_ids and tid2eid must be both present or both absent")

    if torch.compiler.is_compiling():
        placeholder_bias = gating_output if correction_bias is None else correction_bias
        has_bias = correction_bias is not None
        if input_ids is None:
            if _topk_softplus_sqrt_op is None:
                raise _missing_triton_op()
            _topk_softplus_sqrt_op(
                topk_weights,
                topk_indices,
                token_expert_indices,
                gating_output,
                renormalize,
                routed_scaling_factor,
                placeholder_bias,
                has_bias,
            )
            return
        if _topk_hash_softplus_sqrt_op is None:
            raise _missing_triton_op()
        _topk_hash_softplus_sqrt_op(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            placeholder_bias,
            has_bias,
            input_ids,
            tid2eid,
        )
        return

    _eager_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
    )


__all__ = [
    "GROUPED_TOPK_GROUP_JIT",
    "GROUPED_TOPK_SELECT_JIT",
    "GROUPED_TOPK_SPEC",
    "TOPK_HASH_SOFTPLUS_SQRT_JIT",
    "TOPK_HASH_SOFTPLUS_SQRT_SPEC",
    "TOPK_SOFTPLUS_SQRT_JIT",
    "TOPK_SOFTPLUS_SQRT_SPEC",
    "grouped_topk",
    "supports_pt2_moe_routing",
    "topk_softplus_sqrt",
    "uses_common_moe_routing_kernels",
]
