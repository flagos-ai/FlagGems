import torch
import triton
import triton.language as tl


@triton.jit
def replication_pad1d_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    W_in,
    W_out,
    pad_left,
    in_stride_n,
    in_stride_c,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_nc = tl.program_id(axis=1)

    n = pid_nc // C
    c = pid_nc % C

    off_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off_w < W_out

    # Compute clamped source indices for replication pad
    w_in = off_w - pad_left
    w_in = tl.maximum(w_in, 0)
    w_in = tl.minimum(w_in, W_in - 1)

    # Base offsets
    base_in = n.to(tl.int64) * in_stride_n + c.to(tl.int64) * in_stride_c
    base_out = n.to(tl.int64) * out_stride_n + c.to(tl.int64) * out_stride_c

    ptrs_in = in_ptr + base_in + w_in.to(tl.int64) * in_stride_w
    ptrs_out = out_ptr + base_out + off_w.to(tl.int64) * out_stride_w

    x = tl.load(ptrs_in, mask=mask, other=0)
    tl.store(ptrs_out, x, mask=mask)


def _launch_replication_pad1d_kernel(input: torch.Tensor, padding, out: torch.Tensor):
    if not input.is_cuda or not out.is_cuda:
        raise RuntimeError("Triton kernels require CUDA tensors")

    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])
    if left < 0 or right < 0:
        raise ValueError("Padding values must be non-negative for replication_pad1d")

    dim = input.dim()
    if dim not in (2, 3):
        raise ValueError("replication_pad1d expects 2D (C, W) or 3D (N, C, W) input")

    if dim == 3:
        N, C, W_in = input.shape
        B = N
        in_s_n, in_s_c, in_s_w = input.stride()
        out_s_n, out_s_c, out_s_w = out.stride()
        expected_out_shape = (N, C, W_in + left + right)
    else:
        C, W_in = input.shape
        B = 1
        N = 1  # dummy
        in_s_c, in_s_w = input.stride()
        in_s_n = 0
        if out.dim() == 2:
            out_s_c, out_s_w = out.stride()
            out_s_n = 0
        elif out.dim() == 3:
            out_s_n, out_s_c, out_s_w = out.stride()
        else:
            raise ValueError("Output tensor has invalid dimensions")
        expected_out_shape = (C, W_in + left + right)

    W_out = W_in + left + right

    # Validate output shape
    if tuple(out.shape) != expected_out_shape:
        raise ValueError(
            f"Output tensor has incorrect shape. Expected {expected_out_shape}, got {tuple(out.shape)}"
        )

    grid = (triton.cdiv(W_out, 256), B * C)
    replication_pad1d_kernel[grid](
        input,
        out,
        B,
        C,
        W_in,
        W_out,
        left,
        in_s_n if dim == 3 else in_s_n,
        in_s_c,
        in_s_w,
        out_s_n if (dim == 3 or out.dim() == 3) else 0,
        out_s_c,
        out_s_w,
        BLOCK_SIZE=256,
    )
    return out


def replication_pad1d(input: torch.Tensor, padding):
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])
    if input.dim() == 3:
        N, C, W_in = input.shape
        out = torch.empty(
            (N, C, W_in + left + right),
            device=input.device,
            dtype=input.dtype,
            layout=input.layout,
        )
    elif input.dim() == 2:
        C, W_in = input.shape
        out = torch.empty(
            (C, W_in + left + right),
            device=input.device,
            dtype=input.dtype,
            layout=input.layout,
        )
    else:
        raise ValueError("replication_pad1d expects 2D (C, W) or 3D (N, C, W) input")
    return _launch_replication_pad1d_kernel(input, (left, right), out)


def replication_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    left, right = int(padding[0]), int(padding[1])

    # Validate dtype/device
    if out.dtype != input.dtype:
        raise ValueError("Output dtype must match input dtype")
    if out.device != input.device:
        raise ValueError("Output device must match input device")

    return _launch_replication_pad1d_kernel(input, (left, right), out)
