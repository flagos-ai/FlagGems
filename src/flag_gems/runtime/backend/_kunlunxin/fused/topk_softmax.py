import torch
import triton
import triton.language as tl


# NOTE: on the kunlunxin backend the actual implementation lives in the C++
# launcher (third_party/xpu/device/xpu3/launch_extra.cpp ::
# handle_topk_softmax) — it intercepts kernels whose name contains
# "topk_gating_softmax_kernel" via a substring match in
# try_launch_table() and runs the fused
#     sorted_softmax_topk / reduce_sum / broadcast_div / range / transpose
# device path.  This Triton kernel body is therefore a NO-OP stub; only its
# name is meaningful (to match the dispatch table) and its signature must
# carry the parameters the launcher reads from `kernelParams`.
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
    renormalize,  # runtime i32: 0 / 1, consumed by the C++ launcher
    index_ty_signal,  # runtime i32: 0=int32, 1=int64, 2=uint32 (consumed by C++)
    INDEX_TY: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_EXPERTS: tl.constexpr,
):
    # Empty body — the C++ launcher in launch_extra.cpp does all the real
    # work via baidu::xpu::api::softmax + topk + reduce_sum / broadcast_div.
    pid = tl.program_id(0)
    _ = pid  # silence unused-variable diagnostics


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
        index_ty_signal = 0
    elif topk_indices.dtype == torch.int64:
        index_ty = tl.int64
        index_ty_signal = 1
    elif topk_indices.dtype == torch.uint32:
        index_ty = tl.uint32
        index_ty_signal = 2
    else:
        raise TypeError("topk_indices must be int32/int64/uint32")

    # Block sizes are unused by the C++ launcher but still required to keep
    # the Triton constexpr signature stable.
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
        renormalize=1 if renormalize else 0,
        index_ty_signal=index_ty_signal,
        INDEX_TY=index_ty,
        BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
        BLOCK_SIZE_EXPERTS=BLOCK_SIZE_EXPERTS,
        isCloseCoreTiling=True,
    )
