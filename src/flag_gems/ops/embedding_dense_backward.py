import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def _embedding_dense_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_n)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0).to(tl.float32)

    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)


@triton.jit
def _embedding_dense_backward_count_kernel(
    indices_ptr,
    counts_ptr,
    N,
    num_weights,
    padding_idx,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = tl.load(indices_ptr + offs, mask=mask, other=0).to(tl.int32)
    valid = mask & (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
    tl.atomic_add(counts_ptr + idx, 1, mask=valid)


@triton.jit
def _embedding_dense_backward_kernel_scale_by_freq(
    grad_output_ptr,
    indices_ptr,
    counts_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_n).to(tl.int32)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    go_ptrs = grad_output_ptr + pid_n * EMBED_DIM + offs_d
    # go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)
    go = tl.load(go_ptrs, mask=mask_d, other=0.0)

    # cnt = tl.load(counts_ptr + idx, mask=valid, other=1).to(tl.float32)
    cnt = tl.load(counts_ptr + idx, mask=valid, other=1)
    go = go / cnt

    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)


def embedding_dense_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    logger.debug("GEMS: embedding_dense_backward")
    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    if (
        grad_output.device.type != flag_gems.device
        or indices.device.type != flag_gems.device
        or grad_output.device != indices.device
    ):
        raise ValueError(
            f"Inputs must be {flag_gems.device} tensors on the same device."
        )

    device = grad_output.device
    assert (
        grad_output.dim() >= 2
    ), "grad_output must have embedding dimension as the last dim."

    D = grad_output.shape[-1]
    go = grad_output.contiguous().view(-1, D)  # (N, D)
    idx = indices.contiguous().view(-1)
    N = idx.numel()

    assert go.shape[0] == N, "indices number must match grad_output rows."
    grad_weight_fp32 = torch.zeros((num_weights, D), device=device, dtype=torch.float32)

    BLOCK_D = 128
    grid = (N, triton.cdiv(D, BLOCK_D))

    if scale_grad_by_freq:
        counts = torch.zeros((num_weights,), device=device, dtype=torch.int32)
        BLOCK_N = 512
        _embedding_dense_backward_count_kernel[(triton.cdiv(N, BLOCK_N),)](
            idx,
            counts,
            N,
            num_weights,
            padding_idx if padding_idx is not None else -1,
            BLOCK_N=BLOCK_N,
        )

        _embedding_dense_backward_kernel_scale_by_freq[grid](
            go,
            idx,
            counts,
            grad_weight_fp32,
            num_weights,
            padding_idx if padding_idx is not None else -1,
            BLOCK_D=BLOCK_D,
            EMBED_DIM=D,
        )
    else:
        _embedding_dense_backward_kernel[grid](
            go,
            idx,
            grad_weight_fp32,
            num_weights,
            padding_idx if padding_idx is not None else -1,
            BLOCK_D=BLOCK_D,
            EMBED_DIM=D,
        )

    if grad_output.dtype != torch.float32:
        return grad_weight_fp32.to(grad_output.dtype)
    return grad_weight_fp32
