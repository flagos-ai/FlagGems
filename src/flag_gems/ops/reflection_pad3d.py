import logging
import math

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad3d_kernel(
    in_ptr,
    out_ptr,
    B,
    D_in,
    H_in,
    W_in,
    pad_front,
    pad_top,
    pad_left,
    D_out,
    H_out,
    W_out,
    BLOCK_DHW: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Flatten 3D index to 1D for block processing
    offs_n = pid_n * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    # Decode to (d, h, w) coordinates
    d_idx = offs_n // (H_out * W_out)
    hw_idx = offs_n % (H_out * W_out)
    h_idx = hw_idx // W_out
    w_idx = hw_idx % W_out

    mask = (offs_n < D_out * H_out * W_out) & (pid_b < B)

    base_in = pid_b * (D_in * H_in * W_in)
    base_out = pid_b * (D_out * H_out * W_out)

    # Compute reflected indices for depth
    z = d_idx.to(tl.int32) - pad_front
    Dm1 = D_in - 1
    pD = 2 * Dm1
    t_d = tl.abs(z)
    m_d = t_d % pD
    id_ = tl.where(m_d < D_in, m_d, pD - m_d)

    # Compute reflected indices for height
    y = h_idx.to(tl.int32) - pad_top
    Hm1 = H_in - 1
    pH = 2 * Hm1
    t_h = tl.abs(y)
    m_h = t_h % pH
    ih = tl.where(m_h < H_in, m_h, pH - m_h)

    # Compute reflected indices for width
    x = w_idx.to(tl.int32) - pad_left
    Wm1 = W_in - 1
    pW = 2 * Wm1
    t_w = tl.abs(x)
    m_w = t_w % pW
    iw = tl.where(m_w < W_in, m_w, pW - m_w)

    # Load from input and store to output
    in_offs = id_ * (H_in * W_in) + ih * W_in + iw
    vals = tl.load(in_ptr + base_in + in_offs, mask=mask, other=0)
    tl.store(out_ptr + base_out + offs_n, vals, mask=mask)


def launch_reflection_pad3d(input: torch.Tensor, padding, out: torch.Tensor = None):
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 6:
        raise ValueError(
            "padding must be a sequence of length 6: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)"
        )
    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = [
        int(p) for p in padding
    ]

    # Validate padding values
    if (
        pad_left < 0
        or pad_right < 0
        or pad_top < 0
        or pad_bottom < 0
        or pad_front < 0
        or pad_back < 0
    ):
        raise ValueError("padding values must be >= 0")

    # Validate input
    if input.dim() < 4:
        raise ValueError("input must have at least 4 dimensions")
    if input.device.type != flag_gems.device:
        raise ValueError(f"input must be a {flag_gems.device} tensor")

    x = input.contiguous()
    D_in = int(x.shape[-3])
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])

    # Validate reflection padding constraints
    if D_in < 2 or H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 for reflection padding when padding > 0"
        )
    if (
        pad_left >= W_in
        or pad_right >= W_in
        or pad_top >= H_in
        or pad_bottom >= H_in
        or pad_front >= D_in
        or pad_back >= D_in
    ):
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    D_out = D_in + pad_front + pad_back
    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-3]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Handle output tensor
    if out is None:
        out = torch.empty(
            (*leading_shape, D_out, H_out, W_out),
            device=x.device,
            dtype=x.dtype,
        )
    else:
        if out.device.type != flag_gems.device:
            raise ValueError(f"out must be a {flag_gems.device} tensor")
        expected_shape = (*leading_shape, D_out, H_out, W_out)
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

    BLOCK_DHW = 256
    grid = (B, triton.cdiv(D_out * H_out * W_out, BLOCK_DHW))
    reflection_pad3d_kernel[grid](
        x, out, B, D_in, H_in, W_in, pad_front, pad_top, pad_left, D_out, H_out, W_out, BLOCK_DHW=BLOCK_DHW
    )
    return out


def reflection_pad3d(input: torch.Tensor, padding):
    logger.debug("GEMS REFLECTION_PAD3D")
    return launch_reflection_pad3d(input, padding, out=None)


def reflection_pad3d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS REFLECTION_PAD3D_OUT")
    return launch_reflection_pad3d(input, padding, out=out)
