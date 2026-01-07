import torch
import triton
import triton.language as tl


@triton.jit
def t(
    in_ptr,
    out_ptr,
    M,
    N,
    stride_in_0,
    stride_in_1,
    stride_out_0,
    stride_out_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    mask = (offs_m < M) & (offs_n < N)

    offs_m64 = offs_m.to(tl.int64)
    offs_n64 = offs_n.to(tl.int64)

    in_offsets = offs_m64 * stride_in_0 + offs_n64 * stride_in_1
    vals = tl.load(in_ptr + in_offsets, mask=mask)

    out_offsets = offs_n64 * stride_out_0 + offs_m64 * stride_out_1
    tl.store(out_ptr + out_offsets, vals, mask=mask)


_t_kernel = t


def t(*args, **kwargs):
    x = None
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        x = args[0]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        x = kwargs["self"]
    elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
        x = kwargs["input"]
    if x is None:
        raise TypeError("t expects a single Tensor argument")

    if x.dim() == 1:
        return x.clone()
    if x.dim() != 2:
        raise RuntimeError("t: input tensor must be 1D or 2D")

    assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernel"

    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _t_kernel[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        num_warps=4,
    )
    return y
