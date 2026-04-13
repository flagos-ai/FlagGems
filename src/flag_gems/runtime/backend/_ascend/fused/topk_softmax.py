import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.fused.{__name__.split(".")[-1]}')


@libentry()
@triton.jit
def topk_gating_softmax_kernel(
    input_ptr,
    finished_ptr,  # interface reserved, not yet used
    output_ptr,
    indices_ptr,
    source_rows_ptr,
    num_rows,
    k,
    num_experts,
    start_expert,
    end_expert,
    renormalize: tl.constexpr,
    INDEX_TY: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_EXPERTS: tl.constexpr,
):
    """
    Ascend-optimized TopK Gating Softmax kernel.

    Performs softmax on gating logits and selects top-k experts per token.
    Optimized for Ascend NPU with:
    - Core-level tiling to avoid coreDim overflow
    - care_padding=False for better instruction parallelism
    """
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE_ROWS) + pid * BLOCK_SIZE_ROWS
    valid_rows = rows < num_rows

    cols = start_expert + tl.arange(0, BLOCK_SIZE_EXPERTS)
    valid_cols = cols < end_expert

    # Load logits - use care_padding=False for better NPU performance
    logits = tl.load(
        input_ptr + rows[:, None] * num_experts + cols[None, :],
        mask=valid_rows[:, None] & valid_cols[None, :],
        other=-float("inf"),
        care_padding=False,
    ).to(tl.float32)

    # Compute softmax: exp(x - max) / sum(exp(x - max))
    row_max = tl.max(logits, axis=1)[:, None]
    exp_vals = tl.exp(logits - row_max)
    probs = exp_vals / (tl.sum(exp_vals, axis=1)[:, None] + 1e-8)

    # Top-k selection
    selected_sum = tl.zeros([BLOCK_SIZE_ROWS], dtype=tl.float32)

    for ki in range(k):
        # Find max probability and its index
        curr_max, curr_arg = tl.max(probs, axis=1, return_indices=True)

        # Store results
        tl.store(output_ptr + rows * k + ki, curr_max, mask=valid_rows)
        tl.store(indices_ptr + rows * k + ki, curr_arg.to(INDEX_TY), mask=valid_rows)
        tl.store(
            source_rows_ptr + rows * k + ki,
            (ki * num_rows + rows).to(tl.int32),
            mask=valid_rows,
        )

        if renormalize:
            selected_sum += curr_max

        # Mask out selected expert for next iteration
        probs = tl.where(
            cols[None, :] == (curr_arg[:, None] - start_expert),
            -float("inf"),
            probs,
        )

    # Renormalize if requested
    if renormalize:
        norm = selected_sum + 1e-8
        for ki in range(k):
            idx = rows * k + ki
            val = tl.load(output_ptr + idx, mask=valid_rows, care_padding=False)
            tl.store(output_ptr + idx, val / norm, mask=valid_rows)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """
    Ascend-optimized TopK Softmax for MoE gating.

    Computes softmax on gating logits and selects top-k experts.

    Args:
        topk_weights: Output tensor for top-k weights [num_tokens, k]
        topk_indices: Output tensor for top-k expert indices [num_tokens, k]
        token_expert_indices: Output tensor for source row indices [num_tokens, k]
        gating_output: Input gating logits [num_tokens, num_experts]
        renormalize: Whether to renormalize weights to sum to 1
    """
    logger.debug("GEMS_ASCEND TOPK_SOFTMAX")

    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)
    assert topk <= 32, "topk must be <= 32"

    # Determine index type
    if topk_indices.dtype == torch.int32:
        index_ty = tl.int32
    elif topk_indices.dtype == torch.uint32:
        index_ty = tl.uint32
    elif topk_indices.dtype == torch.int64:
        index_ty = tl.int64
    else:
        raise TypeError("topk_indices must be int32/int64/uint32")

    # Calculate block sizes for Ascend NPU
    # BLOCK_SIZE_EXPERTS must be >= num_experts and power of 2, aligned to 32
    BLOCK_SIZE_EXPERTS = triton.next_power_of_2(num_experts)
    BLOCK_SIZE_EXPERTS = ((BLOCK_SIZE_EXPERTS + 31) // 32) * 32
    BLOCK_SIZE_EXPERTS = min(BLOCK_SIZE_EXPERTS, 1024)

    # For Ascend, we need smaller BLOCK_SIZE_ROWS to avoid UB overflow
    if num_experts > 128:
        BLOCK_SIZE_ROWS = 1
    else:
        BLOCK_SIZE_ROWS = max(1, min(16, 1024 // BLOCK_SIZE_EXPERTS))

    # Grid configuration - ensure coreDim <= 65535
    num_blocks = triton.cdiv(num_tokens, BLOCK_SIZE_ROWS)
    if num_blocks > 65535:
        BLOCK_SIZE_ROWS = triton.cdiv(num_tokens, 65535)
        BLOCK_SIZE_ROWS = triton.next_power_of_2(BLOCK_SIZE_ROWS)
        num_blocks = triton.cdiv(num_tokens, BLOCK_SIZE_ROWS)

    grid = (min(num_blocks, 65535),)

    with torch_device_fn.device(gating_output.device):
        topk_gating_softmax_kernel[grid](
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
            BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
            BLOCK_SIZE_EXPERTS=BLOCK_SIZE_EXPERTS,
        )
