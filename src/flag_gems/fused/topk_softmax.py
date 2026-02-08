import torch
import triton
import triton.language as tl


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
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE_ROWS) + pid * BLOCK_SIZE_ROWS
    valid_rows = rows < num_rows

    cols = start_expert + tl.arange(0, BLOCK_SIZE_EXPERTS)
    valid_cols = cols < end_expert

    logits = tl.load(
        input_ptr + rows[:, None] * num_experts + cols[None, :],
        mask=valid_rows[:, None] & valid_cols[None, :],
        other=-float("inf"),
    ).to(tl.float32)

    row_max = tl.max(logits, axis=1)[:, None]
    exp_vals = tl.exp(logits - row_max)
    probs = exp_vals / (tl.sum(exp_vals, axis=1)[:, None] + 1e-8)

    selected_sum = tl.zeros([BLOCK_SIZE_ROWS], dtype=tl.float32)
    for ki in range(k):
        curr_max, curr_arg = tl.max(probs, axis=1, return_indices=True)

        tl.store(output_ptr + rows * k + ki, curr_max, mask=valid_rows)
        tl.store(indices_ptr + rows * k + ki, curr_arg.to(INDEX_TY), mask=valid_rows)
        tl.store(
            source_rows_ptr + rows * k + ki,
            (ki * num_rows + rows).to(tl.int32),
            mask=valid_rows,
        )
        if renormalize:
            selected_sum += curr_max

        probs = tl.where(
            cols[None, :] == (curr_arg[:, None] - start_expert), -float("inf"), probs
        )

    if renormalize:
        norm = selected_sum + 1e-8
        for ki in range(k):
            idx = rows * k + ki
            val = tl.load(output_ptr + idx, mask=valid_rows)
            tl.store(output_ptr + idx, val / norm, mask=valid_rows)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)
    assert topk <= 32

    if topk_indices.dtype == torch.int32:
        index_ty = tl.int32
    elif topk_indices.dtype == torch.uint32:
        index_ty = tl.uint32
    elif topk_indices.dtype == torch.int64:
        index_ty = tl.int64
    else:
        raise TypeError("topk_indices must be int32/int64/uint32")

    max_total_threads = 1024
    BLOCK_SIZE_EXPERTS = ((triton.next_power_of_2(num_experts) + 31) // 32) * 32
    BLOCK_SIZE_EXPERTS = min(BLOCK_SIZE_EXPERTS, 1024)
    BLOCK_SIZE_ROWS = max_total_threads // BLOCK_SIZE_EXPERTS
    BLOCK_SIZE_ROWS = max(BLOCK_SIZE_ROWS, 1)

    # If num_experts > 128, intra-warp shuffling is forced for reduction,
    # which requires the warp layout to be confined to a single row.
    # Consequently, in the TTGIR, the second dimension of warpsPerCTA is fixed to 1.
    if num_experts > 128:
        BLOCK_SIZE_ROWS = 1
        num_warps = 1
    else:
        num_warps = 4

    grid = (triton.cdiv(num_tokens, BLOCK_SIZE_ROWS),)
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
        num_warps=num_warps,
    )
