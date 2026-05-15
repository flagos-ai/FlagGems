import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(
    A_ptr,  # *Pointer* to matrix A [M, N]
    x_ptr,  # *Pointer* to vector x [N]
    y_ptr,  # *Pointer* to output vector y [M]
    M,  # rows of A
    N,  # cols of A (and size of x)
    stride_am,  # stride for A along M (row stride)
    stride_an,  # stride for A along N (col stride)
    stride_xn,  # stride for x along N
    stride_ym,  # stride for y along M
    BLOCK_N: tl.constexpr,  # tile size along N
):
    pid_m = tl.program_id(axis=0)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((), dtype=tl.float32)

    row_ptr = A_ptr + pid_m * stride_am

    for n0 in range(0, N, BLOCK_N):
        idx_n = n0 + offs_n
        mask = idx_n < N
        a = tl.load(row_ptr + idx_n * stride_an, mask=mask, other=0.0)
        x = tl.load(x_ptr + idx_n * stride_xn, mask=mask, other=0.0)
        # accumulate in fp32 for better precision
        acc += tl.sum(a.to(tl.float32) * x.to(tl.float32), axis=0)

    tl.store(y_ptr + pid_m * stride_ym, acc)


def _launch_mv_kernel(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    M, N = A.shape
    assert x.numel() == N
    grid = (M,)
    mv_kernel[grid](
        A,
        x,
        y,
        M,
        N,
        A.stride(0),
        A.stride(1),
        x.stride(0),
        y.stride(0),
        BLOCK_N=256,
        num_warps=4,
        num_stages=2,
    )


def mv(A: torch.Tensor, x: torch.Tensor):
    # Validate inputs
    assert isinstance(A, torch.Tensor) and isinstance(
        x, torch.Tensor
    ), "Inputs must be tensors"
    assert A.ndim == 2 and x.ndim == 1, "mv expects A: 2D tensor and x: 1D tensor"
    assert A.shape[1] == x.shape[0], "Incompatible dimensions for mv"
    assert (
        A.is_cuda and x.is_cuda and A.device == x.device
    ), "All tensors must be on the same CUDA device"

    # Determine output dtype following PyTorch's type promotion
    out_dtype = torch.result_type(A, x)
    M = A.shape[0]
    if M == 0:
        return torch.empty((0,), device=A.device, dtype=out_dtype)

    # Prepare tensors (dtype + contiguous)
    A_ = A.to(out_dtype).contiguous()
    x_ = x.to(out_dtype).contiguous()
    y = torch.empty((M,), device=A.device, dtype=out_dtype)
    y_ = y.contiguous()

    _launch_mv_kernel(A_, x_, y_)

    if y_.data_ptr() != y.data_ptr():
        y.copy_(y_)
    return y


def mv_out(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    # Validate inputs
    assert (
        isinstance(A, torch.Tensor)
        and isinstance(x, torch.Tensor)
        and isinstance(out, torch.Tensor)
    ), "Inputs must be tensors"
    assert (
        A.ndim == 2 and x.ndim == 1 and out.ndim == 1
    ), "Shapes must be A: [M, N], x: [N], out: [M]"
    assert A.shape[1] == x.shape[0], "Incompatible dimensions for mv.out"
    assert out.shape[0] == A.shape[0], "Output shape must match rows of A"
    assert A.is_cuda and x.is_cuda and out.is_cuda, "All tensors must be CUDA tensors"
    assert A.device == x.device == out.device, "All tensors must be on the same device"

    # Execute in the dtype of out (PyTorch .out usually determines dtype by out)
    compute_dtype = out.dtype
    M = A.shape[0]
    if M == 0:
        return out

    A_ = A.to(compute_dtype).contiguous()
    x_ = x.to(compute_dtype).contiguous()

    if out.is_contiguous():
        _launch_mv_kernel(A_, x_, out)
        return out
    else:
        y_tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch_mv_kernel(A_, x_, y_tmp)
        out.copy_(y_tmp)
        return out
