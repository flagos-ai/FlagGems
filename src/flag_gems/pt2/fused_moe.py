# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Transparent PT2 contracts for common FlagGems MoE primitives.

This module covers two existing NVIDIA/common kernels with different execution
contracts:

* ``topk_softmax`` is one direct JITFunction that mutates three caller-owned
  buffers.  Expert count, top-k, index dtype, and ``renormalize`` determine
  guarded constexpr launch metadata; token count remains a runtime grid value.
* ``moe_sum`` is an Autotuner with four original ``BLOCK_SIZE`` configs and
  ``key=["hidden_size", "topk"]``.  The whole Autotuner and its callable META
  grid are passed to ``wrap_triton``; no config is pinned or copied.

Eager execution retains the original public FlagGems launchers.  Compiled
execution replaces only logging/Python launch control with ``triton_op`` and
``wrap_triton`` around the exact same kernel objects.  No compile-only math
kernel or opaque custom-op fallback is defined here.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.fused.moe_sum import moe_sum as _eager_moe_sum
from flag_gems.fused.moe_sum import moe_sum_kernel as MOE_SUM_AUTOTUNER
from flag_gems.fused.topk_softmax import topk_gating_softmax_kernel as TOPK_SOFTMAX_JIT
from flag_gems.fused.topk_softmax import topk_softmax as _eager_topk_softmax
from flag_gems.pt2.manifest import CompileKind, CompileOpSpec, register_compile_spec

_HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(
    torch.library, "wrap_triton"
)
_TOPK_INDEX_DTYPES = (torch.int32, torch.uint32, torch.int64)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def supports_pt2_fused_moe() -> bool:
    """Return whether transparent Triton operator APIs are available."""

    return _HAS_TRITON_OP


def _check_same_device(reference: torch.Tensor, other: torch.Tensor) -> None:
    torch._check(other.device == reference.device)


def _check_topk_contract(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate the actual dense ABI used by the original top-k kernel."""

    torch._check(gating_output.ndim == 2)
    torch._check(topk_weights.ndim == 2)
    torch._check(topk_indices.ndim == 2)
    torch._check(token_expert_indices.ndim == 2)

    num_tokens = gating_output.shape[0]
    num_experts = gating_output.shape[1]
    topk = topk_weights.shape[1]
    torch._check(num_experts > 0)
    torch._check(num_experts <= 1024)
    torch._check(topk > 0)
    torch._check(topk <= 32)
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

    # The original kernel does not accept strides and computes all addresses
    # as row * dense_width + column.  Reject layouts it cannot represent.
    torch._check(gating_output.is_contiguous())
    torch._check(topk_weights.is_contiguous())
    torch._check(topk_indices.is_contiguous())
    torch._check(token_expert_indices.is_contiguous())
    return num_tokens, num_experts, topk


def _index_triton_dtype(dtype: torch.dtype):
    if dtype == torch.int32:
        return tl.int32
    if dtype == torch.uint32:
        return tl.uint32
    if dtype == torch.int64:
        return tl.int64
    raise TypeError("topk_indices must be int32/int64/uint32")


def _check_moe_sum_contract(
    input: torch.Tensor, output: torch.Tensor
) -> tuple[int, int, int]:
    """Validate the strided-row, contiguous-hidden ABI of ``moe_sum``."""

    torch._check(input.ndim == 3)
    torch._check(output.ndim == 2)
    num_tokens, topk, hidden_size = input.shape
    torch._check(topk > 0)
    torch._check(hidden_size > 0)
    torch._check(output.shape[0] == num_tokens)
    torch._check(output.shape[1] == hidden_size)
    torch._check(input.dtype in _FLOAT_DTYPES)
    torch._check(output.dtype == input.dtype)
    _check_same_device(input, output)

    # input_stride_hidden/output_stride_hidden are present in the Python ABI,
    # but this kernel addresses hidden elements with raw offsets.  Only a
    # unit-stride hidden dimension is semantically supported.
    torch._check(input.stride(2) == 1)
    torch._check(output.stride(1) == 1)
    return num_tokens, topk, hidden_size


if _HAS_TRITON_OP:

    @torch.library.triton_op(
        "flag_gems_pt2::topk_softmax",
        mutates_args={
            "topk_weights",
            "topk_indices",
            "token_expert_indices",
        },
    )
    def _topk_softmax_op(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
    ) -> None:
        num_tokens, num_experts, topk = _check_topk_contract(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
        )
        if num_tokens == 0:
            return

        index_ty = _index_triton_dtype(topk_indices.dtype)
        block_size_experts = ((triton.next_power_of_2(num_experts) + 31) // 32) * 32
        block_size_experts = min(block_size_experts, 1024)
        block_size_rows = max(1024 // block_size_experts, 1)
        if num_experts > 128:
            block_size_rows = 1
            num_warps = 1
        else:
            num_warps = 4
        grid = (triton.cdiv(num_tokens, block_size_rows),)

        torch.library.wrap_triton(TOPK_SOFTMAX_JIT)[grid](
            input_ptr=gating_output,
            finished_ptr=None,
            output_ptr=topk_weights,
            indices_ptr=topk_indices,
            source_rows_ptr=token_expert_indices,
            num_rows=num_tokens,
            k=topk,
            num_experts=num_experts,
            start_expert=0,
            end_expert=num_experts,
            renormalize=renormalize,
            INDEX_TY=index_ty,
            BLOCK_SIZE_ROWS=block_size_rows,
            BLOCK_SIZE_EXPERTS=block_size_experts,
            num_warps=num_warps,
        )

    @torch.library.triton_op("flag_gems_pt2::moe_sum", mutates_args={"output"})
    def _moe_sum_op(input: torch.Tensor, output: torch.Tensor) -> None:
        num_tokens, topk, hidden_size = _check_moe_sum_contract(input, output)
        if num_tokens == 0:
            return

        grid = lambda meta: (
            num_tokens,
            triton.cdiv(hidden_size, meta["BLOCK_SIZE"]),
        )
        torch.library.wrap_triton(MOE_SUM_AUTOTUNER)[grid](
            input,
            output,
            num_tokens,
            topk,
            hidden_size,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            output.stride(0),
            output.stride(1),
        )

else:
    _topk_softmax_op = None
    _moe_sum_op = None


_TRANSPARENT_REQUIRES = (
    "torch.library.triton_op",
    "torch.library.wrap_triton",
    "NVIDIA/common FlagGems kernel identity",
)

TOPK_SOFTMAX_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::topk_softmax",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=("flag_gems.fused.topk_softmax.topk_gating_softmax_kernel"),
        mutates_args=(
            "topk_weights",
            "topk_indices",
            "token_expert_indices",
        ),
        dynamic_dims=("num_tokens",),
        requires=_TRANSPARENT_REQUIRES,
    )
)

MOE_SUM_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::moe_sum",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel="flag_gems.fused.moe_sum.moe_sum_kernel (Autotuner)",
        mutates_args=("output",),
        dynamic_dims=("num_tokens",),
        requires=_TRANSPARENT_REQUIRES,
    )
)


def _missing_triton_op() -> RuntimeError:
    return RuntimeError(
        "This Torch build lacks triton_op/wrap_triton; the transparent "
        "FlagGems MoE primitive PT2 contracts are unavailable"
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """Run the original top-k softmax kernel in eager or compiled mode."""

    if torch.compiler.is_compiling():
        if _topk_softmax_op is None:
            raise _missing_triton_op()
        _topk_softmax_op(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
        return
    _eager_topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    """Run the original autotuned MoE reduction in eager or compiled mode."""

    if torch.compiler.is_compiling():
        if _moe_sum_op is None:
            raise _missing_triton_op()
        _moe_sum_op(input, output)
        return
    _eager_moe_sum(input, output)


__all__ = [
    "MOE_SUM_AUTOTUNER",
    "MOE_SUM_SPEC",
    "TOPK_SOFTMAX_JIT",
    "TOPK_SOFTMAX_SPEC",
    "moe_sum",
    "supports_pt2_fused_moe",
    "topk_softmax",
]
