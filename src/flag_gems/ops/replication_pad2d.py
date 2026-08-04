import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("replication_pad2d"),
    key=["H_out", "W_out"],
)
@triton.jit
def replication_pad2d_kernel(
    x_ptr,
    out_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    pad_l,
    pad_t,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    C,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    c_idx = pid_nc % C
    n_idx = pid_nc // C

    x_base_ptr = x_ptr + n_idx * stride_xn + c_idx * stride_xc
    out_base_ptr = out_ptr + n_idx * stride_on + c_idx * stride_oc

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Compute clamped source indices for replication pad
    iy = offs_h - pad_t
    iy = tl.where(iy < 0, 0, iy)
    iy = tl.where(iy > H_in - 1, H_in - 1, iy)

    ix = offs_w - pad_l
    ix = tl.where(ix < 0, 0, ix)
    ix = tl.where(ix > W_in - 1, W_in - 1, ix)

    x_offset = iy[:, None] * stride_xh + ix[None, :] * stride_xw
    out_offset = offs_h[:, None] * stride_oh + offs_w[None, :] * stride_ow

    mask = (offs_h[:, None] < H_out) & (offs_w[None, :] < W_out)

    vals = tl.load(x_base_ptr + x_offset, mask=mask)
    tl.store(out_base_ptr + out_offset, vals, mask=mask)


def replication_pad2d(x, padding):
    logger.debug("GEMS REPLICATION_PAD2D")
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = padding
    else:
        pad_l, pad_r, pad_t, pad_b = padding

    N, C, H_in, W_in = x.shape
    H_out = H_in + pad_t + pad_b
    W_out = W_in + pad_l + pad_r

    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(W_out, META["BLOCK_W"]),
        triton.cdiv(H_out, META["BLOCK_H"]),
        N * C,
    )

    replication_pad2d_kernel[grid](
        x,
        out,
        H_in,
        W_in,
        H_out,
        W_out,
        pad_l,
        pad_t,
        *x.stride(),
        *out.stride(),
        C,
    )

    return out


def replication_pad2d_out(x, padding, out):
    logger.debug("GEMS REPLICATION_PAD2D_OUT")
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = padding
    else:
        pad_l, pad_r, pad_t, pad_b = padding

    N, C, H_in, W_in = x.shape
    H_out = H_in + pad_t + pad_b
    W_out = W_in + pad_l + pad_r

    expected_shape = (N, C, H_out, W_out)
    if tuple(out.shape) != expected_shape:
        raise ValueError(
            f"Output tensor has incorrect shape. Expected {expected_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != x.dtype:
        raise ValueError("Output dtype must match input dtype")
    if out.device != x.device:
        raise ValueError("Output device must match input device")

    grid = lambda META: (
        triton.cdiv(W_out, META["BLOCK_W"]),
        triton.cdiv(H_out, META["BLOCK_H"]),
        N * C,
    )

    replication_pad2d_kernel[grid](
        x,
        out,
        H_in,
        W_in,
        H_out,
        W_out,
        pad_l,
        pad_t,
        *x.stride(),
        *out.stride(),
        C,
    )

    return out
