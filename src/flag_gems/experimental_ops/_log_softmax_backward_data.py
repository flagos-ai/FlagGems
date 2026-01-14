import math

import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax_bwd_kernel(
    grad_ptr,  # pointer to grad_output (dL/dy)
    y_logsm_ptr,  # pointer to output of log_softmax (y = log_softmax(x))
    grad_in_ptr,  # pointer to grad_input (dL/dx)
    M,  # number of rows (product of all dims except reduction dim)
    K,  # length of the reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * K
    offs = tl.arange(0, BLOCK_SIZE)

    # First pass: compute s = sum_j grad_output[j] along dim
    s = tl.zeros((), dtype=tl.float32)
    num_chunks = tl.cdiv(K, BLOCK_SIZE)
    for chunk in range(0, num_chunks):
        cols = chunk * BLOCK_SIZE + offs
        mask = cols < K
        go_chunk = tl.load(grad_ptr + row_start + cols, mask=mask, other=0.0)
        go32 = go_chunk.to(tl.float32)
        s += tl.sum(go32, axis=0)

    # Second pass: grad_input = grad_output - exp(output) * s
    for chunk in range(0, num_chunks):
        cols = chunk * BLOCK_SIZE + offs
        mask = cols < K
        go_chunk = tl.load(grad_ptr + row_start + cols, mask=mask, other=0.0)
        go32 = go_chunk.to(tl.float32)
        y_chunk = tl.load(y_logsm_ptr + row_start + cols, mask=mask, other=0.0)
        y32 = y_chunk.to(tl.float32)
        sm = tl.exp(y32)
        gi32 = go32 - sm * s
        gi = gi32.to(go_chunk.dtype)
        tl.store(grad_in_ptr + row_start + cols, gi, mask=mask)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    return dim


def _choose_block_size(K: int) -> int:
    if K <= 1:
        return 1
    bs = 1 << (int(math.ceil(math.log2(K))))
    return min(1024, max(1, bs))


def _log_softmax_backward_data_impl(
    grad_output: torch.Tensor, output: torch.Tensor, dim: int
):
    assert (
        grad_output.shape == output.shape
    ), "grad_output and output must have the same shape"
    assert (
        grad_output.device.type == "cuda" and output.device.type == "cuda"
    ), "Inputs must be CUDA tensors"
    assert grad_output.device == output.device, "Inputs must be on the same device"
    assert grad_output.dtype == output.dtype, "Inputs must have the same dtype"
    assert (
        grad_output.is_contiguous(memory_format=torch.contiguous_format)
        == grad_output.is_contiguous()
    ), "Unsupported memory format"

    if grad_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        # Fallback for unsupported dtype (e.g., float64), compute via PyTorch
        # grad_input = grad_output - exp(output) * sum(grad_output, dim=dim, keepdim=True)
        s = grad_output.sum(dim=dim, keepdim=True)
        return grad_output - output.exp() * s

    dim = _normalize_dim(dim, grad_output.ndim)

    # Move reduction dim to the last for contiguous 2D layout [M, K]
    go_last = torch.movedim(grad_output, dim, -1).contiguous()
    y_last = torch.movedim(output, dim, -1).contiguous()
    K = go_last.shape[-1]
    if go_last.numel() == 0 or K == 0:
        return grad_output.clone()

    M = go_last.numel() // K

    go_2d = go_last.view(M, K)
    y_2d = y_last.view(M, K)
    gi_2d = torch.empty_like(go_2d)

    BLOCK_SIZE = _choose_block_size(K)
    grid = (M,)

    _log_softmax_bwd_kernel[grid](
        go_2d,
        y_2d,
        gi_2d,
        M,
        K,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    gi_last = gi_2d.view_as(go_last)
    grad_input = torch.movedim(gi_last, -1, dim)
    return grad_input


def _log_softmax_backward_data(
    grad_output: torch.Tensor, output: torch.Tensor, dim: int, input_dtype: torch.dtype
):
    return _log_softmax_backward_data_impl(grad_output, output, dim)


def _log_softmax_backward_data_out(
    grad_output: torch.Tensor,
    output: torch.Tensor,
    dim: int,
    input_dtype: torch.dtype,
    out: torch.Tensor,
):
    res = _log_softmax_backward_data_impl(grad_output, output, dim)
    out.copy_(res)
    return out
