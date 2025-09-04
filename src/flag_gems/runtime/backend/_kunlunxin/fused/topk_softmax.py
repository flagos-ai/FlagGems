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
    )

    row_max = tl.max(logits, axis=1)[:, None]
    exp_vals = tl.exp(logits - row_max)
    probs = exp_vals / (tl.sum(exp_vals, axis=1)[:, None] + 1e-8)

    for ki in range(k):
        curr_max = tl.max(probs, axis=1)
        curr_arg = tl.argmax(probs, axis=1) + start_expert

        tl.store(output_ptr + rows * k + ki, curr_max, mask=valid_rows)
        tl.store(indices_ptr + rows * k + ki, curr_arg.to(INDEX_TY), mask=valid_rows)
        tl.store(
            source_rows_ptr + rows * k + ki,
            (ki * num_rows + rows).to(tl.int32),
            mask=valid_rows,
        )

        probs = tl.where(
            cols[None, :] == (curr_arg[:, None] - start_expert), -float("inf"), probs
        )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> None:
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)
    assert topk <= 32

    if topk_indices.dtype == torch.int32:
        index_ty = tl.int32
    # elif topk_indices.dtype == torch.uint32:
    #     index_ty = tl.uint32
    elif topk_indices.dtype == torch.int64:
        index_ty = tl.int64
    else:
        raise TypeError("topk_indices must be int32/int64/uint32")

    max_total_threads = 1024
    BLOCK_SIZE_EXPERTS = ((triton.next_power_of_2(num_experts) + 31) // 32) * 32
    BLOCK_SIZE_EXPERTS = min(BLOCK_SIZE_EXPERTS, 1024)
    BLOCK_SIZE_ROWS = max_total_threads // BLOCK_SIZE_EXPERTS
    BLOCK_SIZE_ROWS = max(BLOCK_SIZE_ROWS, 1)

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
        INDEX_TY=index_ty,
        BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
        BLOCK_SIZE_EXPERTS=BLOCK_SIZE_EXPERTS,
        isCloseCoreTiling=True,
    )
