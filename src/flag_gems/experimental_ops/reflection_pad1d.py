import math

import torch
import triton
import triton.language as tl


@triton.jit
def _reflection_pad1d_kernel(
    in_ptr, out_ptr, B, W_in, pad_left, W_out, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_w < W_out) & (pid_b < B)

    base_in = pid_b * W_in
    base_out = pid_b * W_out

    # Compute reflected indices
    x = offs_w.to(tl.int32) - pad_left  # shift by left pad
    Wm1 = W_in - 1
    p = 2 * Wm1  # period for reflection; guaranteed > 0 when this kernel is used

    t = tl.abs(x)
    m = t % p
    iw = tl.where(m < W_in, m, p - m)

    vals = tl.load(in_ptr + base_in + iw, mask=mask, other=0)
    tl.store(out_ptr + base_out + offs_w, vals, mask=mask)


@triton.jit
def _copy_rows_kernel(in_ptr, out_ptr, B, W, BLOCK_W: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_w < W) & (pid_b < B)

    base = pid_b * W
    vals = tl.load(in_ptr + base + offs_w, mask=mask, other=0)
    tl.store(out_ptr + base + offs_w, vals, mask=mask)


def _launch_reflection_pad1d(input: torch.Tensor, padding, out: torch.Tensor = None):
    if not isinstance(padding, (list, tuple)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of length 2: (pad_left, pad_right)"
        )
    pad_left, pad_right = int(padding[0]), int(padding[1])
    if pad_left < 0 or pad_right < 0:
        raise ValueError("padding values must be >= 0")
    if input.dim() < 1:
        raise ValueError("input must have at least 1 dimension")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")

    x = input.contiguous()
    W_in = int(x.shape[-1])
    if W_in <= 0:
        raise ValueError("last dimension (width) must be > 0")

    W_out = W_in + pad_left + pad_right
    leading_shape = x.shape[:-1]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    if out is None:
        out = torch.empty((*leading_shape, W_out), device=x.device, dtype=x.dtype)
    else:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        expected_shape = (*leading_shape, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    # No padding: just copy
    if pad_left == 0 and pad_right == 0:
        if W_out != W_in:
            raise RuntimeError(
                "Internal error: W_out should equal W_in when no padding"
            )
        grid = (B, triton.cdiv(W_in, 256))
        _copy_rows_kernel[grid](x, out, B, W_in, BLOCK_W=256)
        return out

    # Validate reflection padding constraints
    if W_in < 2:
        raise ValueError(
            "input width must be at least 2 for reflection padding when padding > 0"
        )
    if pad_left >= W_in or pad_right >= W_in:
        raise ValueError(
            "padding values must be less than the input width for reflection padding"
        )

    grid = (B, triton.cdiv(W_out, 256))
    _reflection_pad1d_kernel[grid](x, out, B, W_in, pad_left, W_out, BLOCK_W=256)
    return out


def reflection_pad1d(input: torch.Tensor, padding):
    return _launch_reflection_pad1d(input, padding, out=None)


def reflection_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    return _launch_reflection_pad1d(input, padding, out=out)
