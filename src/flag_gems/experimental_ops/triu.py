import torch
import triton
import triton.language as tl


@triton.jit
def triu_kernel(
    in_ptr, out_ptr,
    M, N, B,  # matrix rows, cols, number of batches
    stride_in_b, stride_in_m, stride_in_n,
    stride_out_b, stride_out_m, stride_out_n,
    diagonal: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (row[:, None] < M) & (col[None, :] < N)

    # keep if col - row >= diagonal
    keep = (col[None, :] - row[:, None]) >= diagonal

    row_i64 = row[:, None].to(tl.int64)
    col_i64 = col[None, :].to(tl.int64)

    base_in = in_ptr + pid_b.to(tl.int64) * stride_in_b
    base_out = out_ptr + pid_b.to(tl.int64) * stride_out_b

    in_offsets = row_i64 * stride_in_m + col_i64 * stride_in_n
    out_offsets = row_i64 * stride_out_m + col_i64 * stride_out_n

    vals = tl.load(base_in + in_offsets, mask=mask & keep, other=0)
    tl.store(base_out + out_offsets, vals, mask=mask)


def _check_supported_dtype(t: torch.Tensor):
    if t.dtype in (torch.complex64, torch.complex128, torch.complex32 if hasattr(torch, "complex32") else None):
        raise TypeError("Complex dtypes are not supported by this Triton triu implementation.")


def _launch_triu_kernel(inp: torch.Tensor, out: torch.Tensor, diagonal: int):
    assert inp.is_cuda and out.is_cuda, "Input and output must be CUDA tensors"
    assert inp.dtype == out.dtype, "Input and output dtypes must match"
    assert inp.device == out.device, "Input and output must be on the same device"
    _check_supported_dtype(inp)

    ndim = inp.dim()
    assert ndim >= 2, "triu expects input with at least 2 dimensions"

    M = inp.shape[-2]
    N = inp.shape[-1]
    batch_shape = inp.shape[:-2]
    B = 1
    for s in batch_shape:
        B *= s

    # Ensure contiguous layout for simplicity
    inp_c = inp.contiguous()
    out_c = out.contiguous()

    # Strides as int64
    stride_in_n = inp_c.stride(-1)
    stride_in_m = inp_c.stride(-2)
    stride_out_n = out_c.stride(-1)
    stride_out_m = out_c.stride(-2)

    # Batch stride: distance between consecutive matrices in flattened batch
    stride_in_b = M * stride_in_m if len(batch_shape) == 0 else inp_c.stride(-3) * inp_c.size(-3) if ndim > 2 else M * stride_in_m
    stride_out_b = M * stride_out_m if len(batch_shape) == 0 else out_c.stride(-3) * out_c.size(-3) if ndim > 2 else M * stride_out_m

    # For fully contiguous tensors, the above may not equal true batch stride for high dims.
    # Since we used .contiguous(), we can simply set:
    if inp_c.is_contiguous():
        stride_in_n = 1
        stride_in_m = N
        stride_in_b = M * N
    if out_c.is_contiguous():
        stride_out_n = 1
        stride_out_m = N
        stride_out_b = M * N

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B)

    triu_kernel[grid](
        inp_c, out_c,
        M, N, B,
        stride_in_b, stride_in_m, stride_in_n,
        stride_out_b, stride_out_m, stride_out_n,
        diagonal=diagonal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    if out.data_ptr() != out_c.data_ptr():
        out.copy_(out_c)


def triu(input: torch.Tensor, diagonal: int = 0):
    """
    Wrapper for ATen op: ('triu', <Autograd.disable: False>)
    """
    out = torch.empty_like(input)
    _launch_triu_kernel(input, out, diagonal)
    return out


def triu_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    """
    Wrapper for ATen op: ('triu.out', <Autograd.disable: False>)
    """
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError(f"out tensor must have the same shape as input, got {out.shape} vs {input.shape}")
        if out.dtype != input.dtype:
            raise TypeError(f"out dtype must match input dtype, got {out.dtype} vs {input.dtype}")
        if not out.is_cuda or out.device != input.device:
            raise ValueError("out must be a CUDA tensor on the same device as input")
    _launch_triu_kernel(input, out, diagonal)
    return out