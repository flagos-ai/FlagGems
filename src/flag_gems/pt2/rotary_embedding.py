# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PT2 contract for the existing FlagGems rotary embedding Triton kernel."""

import torch
import triton

from flag_gems.fused.rotary_embedding import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_inplace_kernel,
)
from flag_gems.pt2.manifest import CompileKind, CompileOpSpec, register_compile_spec

_RAW_INPLACE_KERNEL = apply_rotary_pos_emb_inplace_kernel.jit_function
_HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(
    torch.library, "wrap_triton"
)


def supports_pt2_triton() -> bool:
    """Return whether this Torch build exposes the required PT2 Triton APIs."""

    return _HAS_TRITON_OP


if _HAS_TRITON_OP:

    @torch.library.triton_op(
        "flag_gems_pt2::rotary_embedding_inplace",
        mutates_args={"query", "key"},
    )
    def _rotary_embedding_inplace_op(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool,
    ) -> None:
        """Launch the original FlagGems JITFunction with a traceable call."""

        n_tokens = query.shape[0]
        head_dim = query.shape[-1]
        padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

        torch.library.wrap_triton(_RAW_INPLACE_KERNEL)[(n_tokens,)](
            query,
            key,
            cos,
            sin,
            position_ids,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            position_ids.stride(0),
            cos.stride(0),
            sin.stride(0),
            0,  # seq_len is unused when position_ids is provided.
            query.shape[-2],
            key.shape[-2],
            head_dim,
            padded_head_dim,
            rotary_interleaved,
            MAX_POSITION_EMBEDDINGS=cos.shape[0],
        )

else:
    _rotary_embedding_inplace_op = None


ROTARY_EMBEDDING_INPLACE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::rotary_embedding_inplace",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.rotary_embedding."
            "apply_rotary_pos_emb_inplace_kernel.jit_function"
        ),
        mutates_args=("query", "key"),
        dynamic_dims=("n_tokens",),
        requires=("torch.library.triton_op", "torch.library.wrap_triton"),
    )
)


def rotary_embedding_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
) -> None:
    """Use one kernel body in eager and compiled execution.

    Supported PT2 builds call the registered ``triton_op`` in both modes.  An
    older vendor Torch build may still use the original LibEntry eager launch,
    but compilation fails explicitly instead of substituting a native kernel.
    """

    if _rotary_embedding_inplace_op is not None:
        _rotary_embedding_inplace_op(
            query, key, cos, sin, position_ids, rotary_interleaved
        )
        return

    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        raise RuntimeError(
            "This Torch build does not provide torch.library.triton_op and "
            "torch.library.wrap_triton; FlagGems RoPE cannot enter a PT2 graph"
        )

    # Compatibility for eager-only vendor Torch builds.  This is still the
    # exact same Triton kernel object; it is not a PyTorch-native fallback.
    apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=True,
    )


__all__ = [
    "ROTARY_EMBEDDING_INPLACE_SPEC",
    "rotary_embedding_inplace",
    "supports_pt2_triton",
]
