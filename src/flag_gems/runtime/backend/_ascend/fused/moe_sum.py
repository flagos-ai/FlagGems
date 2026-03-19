import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.fused.{__name__.split(".")[-1]}')


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("moe_sum"),
    key=["hidden_size", "topk"],
)
@triton.jit
def moe_sum_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    output_stride_token,
    IS_CONTIGUOUS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    """
    Ascend-optimized MoE sum kernel.

    Optimization Round 5:
    - Manual loop unrolling hints for common topk values
    - Reduced loop overhead for small topk
    - Vectorized accumulation pattern
    """
    pid = tle.program_id(0)

    # Task partition
    num_hidden_blocks = tl.cdiv(hidden_size, BLOCK_SIZE)
    token_idx = pid // num_hidden_blocks
    block_idx = pid % num_hidden_blocks

    if token_idx >= num_tokens:
        return

    hidden_base = block_idx * BLOCK_SIZE

    if IS_CONTIGUOUS:
        # Contiguous tensor path - optimized for common case
        input_token_offset = token_idx * topk * hidden_size
        output_token_offset = token_idx * hidden_size

        for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            h_offset = hidden_base + sub_idx
            h_indices = h_offset + tl.arange(0, BLOCK_SIZE_SUB)
            valid_mask = h_indices < hidden_size

            # Initialize accumulator
            result = tl.zeros((BLOCK_SIZE_SUB,), dtype=tl.float32)

            # Compute base pointer for expert 0
            base = input_ptr + input_token_offset + h_indices
            expert_stride = hidden_size

            # Accumulate - compiler unrolls for small constant topk
            # For topk=2,4,8 this is fully unrolled
            for k in range(topk):
                val = tl.load(
                    base + k * expert_stride,
                    mask=valid_mask,
                    other=0.0,
                    care_padding=False,
                )
                result += val.to(tl.float32)

            # Store
            out_ptr = output_ptr + output_token_offset + h_indices
            tl.store(out_ptr, result.to(output_ptr.dtype.element_ty), mask=valid_mask)

    else:
        # Non-contiguous path
        input_base = input_ptr + token_idx * input_stride_token
        output_base = output_ptr + token_idx * output_stride_token

        for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            h_offset = hidden_base + sub_idx
            h_indices = h_offset + tl.arange(0, BLOCK_SIZE_SUB)
            valid_mask = h_indices < hidden_size

            result = tl.zeros((BLOCK_SIZE_SUB,), dtype=tl.float32)

            for k in range(topk):
                ptr = input_base + k * input_stride_topk + h_indices
                val = tl.load(ptr, mask=valid_mask, other=0.0, care_padding=False)
                result += val.to(tl.float32)

            tl.store(
                output_base + h_indices,
                result.to(output_ptr.dtype.element_ty),
                mask=valid_mask,
            )


# Specialized kernel for topk=2 (most common in MoE)
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("moe_sum"),
    key=["hidden_size"],
)
@triton.jit
def moe_sum_kernel_topk2(
    input_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    """Specialized kernel for topk=2 with fully unrolled expert loop."""
    pid = tle.program_id(0)

    num_hidden_blocks = tl.cdiv(hidden_size, BLOCK_SIZE)
    token_idx = pid // num_hidden_blocks
    block_idx = pid % num_hidden_blocks

    if token_idx >= num_tokens:
        return

    hidden_base = block_idx * BLOCK_SIZE
    input_token_offset = token_idx * 2 * hidden_size
    output_token_offset = token_idx * hidden_size

    for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        h_offset = hidden_base + sub_idx
        h_indices = h_offset + tl.arange(0, BLOCK_SIZE_SUB)
        valid_mask = h_indices < hidden_size

        base = input_ptr + input_token_offset + h_indices

        # Fully unrolled for topk=2
        val0 = tl.load(base, mask=valid_mask, other=0.0, care_padding=False)
        val1 = tl.load(base + hidden_size, mask=valid_mask, other=0.0, care_padding=False)

        result = val0.to(tl.float32) + val1.to(tl.float32)

        out_ptr = output_ptr + output_token_offset + h_indices
        tl.store(out_ptr, result.to(output_ptr.dtype.element_ty), mask=valid_mask)


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
):
    """
    MoE sum operation optimized for Ascend NPU.

    Sums over the expert dimension (dim=1).
    Input shape: (num_tokens, topk, hidden_size)
    Output shape: (num_tokens, hidden_size)
    """
    logger.debug("GEMS_ASCEND MOE_SUM")

    num_tokens, topk, hidden_size = input.shape

    # Get strides
    in_s0, in_s1, in_s2 = input.stride()
    out_s0, out_s1 = output.stride()

    # Check contiguous pattern
    is_contiguous = (
        in_s2 == 1
        and in_s1 == hidden_size
        and in_s0 == topk * hidden_size
        and out_s1 == 1
        and out_s0 == hidden_size
    )

    def grid(meta):
        n_blocks = triton.cdiv(hidden_size, meta["BLOCK_SIZE"])
        total = num_tokens * n_blocks
        return (min(total, 65535),)

    with torch_device_fn.device(input.device):
        # Use specialized kernel for topk=2 (most common case)
        if topk == 2 and is_contiguous:
            moe_sum_kernel_topk2[grid](
                input,
                output,
                num_tokens,
                hidden_size,
            )
        else:
            moe_sum_kernel[grid](
                input,
                output,
                num_tokens,
                topk,
                hidden_size,
                in_s0,
                in_s1,
                out_s0,
                IS_CONTIGUOUS=is_contiguous,
            )
