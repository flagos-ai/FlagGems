# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of reflection_pad3d / reflection_pad3d_out.
#
# Root cause: the generic kernel (flag_gems/ops/reflection_pad3d.py) computes the
# reflected input index with a runtime modulo:
#     m = tl.abs(coord) % (2*(dim-1)); idx = where(m<dim, m, 2*(dim-1)-m)
# On XPU that runtime `%` miscompiles at the reflection boundaries (0.5% of
# elements wrong, max abs diff ~3.1) -- the same modulo that the reflection_pad2d
# XPU override already had to remove (see reflection_pad2d.py header).
#
# Fix: because the host validates pad < dim on every axis, |coord| never exceeds
# one period 2*(dim-1), so a SINGLE-period `abs + where` is mathematically exact
# and needs no modulo. Flatten (b, d, h, w) into one linear output index, decode
# with div/mod, gather the reflected input, mask-based contiguous store.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad3d_kernel(
    in_ptr,
    out_ptr,
    D_in,
    H_in,
    W_in,
    pad_f,
    pad_t,
    pad_l,
    D_out,
    H_out,
    W_out,
    DHW_out,
    DHW_in,
    HW_out,
    HW_in,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # Decode flat output index -> (batch, d_out, h_out, w_out)
    b = o // DHW_out
    rem = o % DHW_out
    d_idx = rem // HW_out
    rem2 = rem % HW_out
    h_idx = rem2 // W_out
    w_idx = rem2 % W_out

    # Single-period reflection (pad < dim validated on host -> no modulo needed).
    z = d_idx.to(tl.int32) - pad_f
    pD = 2 * (D_in - 1)
    t_d = tl.abs(z)
    id_ = tl.where(t_d < D_in, t_d, pD - t_d)

    y = h_idx.to(tl.int32) - pad_t
    pH = 2 * (H_in - 1)
    t_h = tl.abs(y)
    ih = tl.where(t_h < H_in, t_h, pH - t_h)

    x = w_idx.to(tl.int32) - pad_l
    pW = 2 * (W_in - 1)
    t_w = tl.abs(x)
    iw = tl.where(t_w < W_in, t_w, pW - t_w)

    in_offs = b * DHW_in + id_ * HW_in + ih * W_in + iw
    vals = tl.load(in_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total
    vals = tl.load(in_ptr + o, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


def launch_reflection_pad3d(input: torch.Tensor, padding, out: torch.Tensor = None):
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 6:
        raise ValueError(
            "padding must be a sequence of length 6: "
            "(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)"
        )
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = [int(p) for p in padding]

    if min(pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba) < 0:
        raise ValueError("padding values must be >= 0")

    # 4D input (C, D, H, W) -> unsqueeze batch.
    is_4d = input.dim() == 4
    if is_4d:
        input = input.unsqueeze(0)
    if input.dim() < 5:
        raise ValueError("input must have at least 5 dimensions")

    x = input.contiguous()
    N, C, D_in, H_in, W_in = x.shape

    if D_in < 2 or H_in < 2 or W_in < 2:
        raise ValueError("input spatial dimensions must be at least 2")
    if (
        pad_l >= W_in
        or pad_r >= W_in
        or pad_t >= H_in
        or pad_b >= H_in
        or pad_f >= D_in
        or pad_ba >= D_in
    ):
        raise ValueError(
            "padding values must be less than the input spatial dimensions"
        )

    D_out = D_in + pad_f + pad_ba
    H_out = H_in + pad_t + pad_b
    W_out = W_in + pad_l + pad_r

    B = N * C
    if out is None:
        out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    else:
        expected_shape = (N, C, D_out, H_out, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(f"out dtype {out.dtype} != input dtype {x.dtype}")
        out = out.contiguous()

    BLOCK = 1024
    # No padding: flat contiguous copy.
    if pad_l == pad_r == pad_t == pad_b == pad_f == pad_ba == 0:
        total = B * D_in * H_in * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out.squeeze(0) if is_4d else out

    HW_out = H_out * W_out
    HW_in = H_in * W_in
    DHW_out = D_out * HW_out
    DHW_in = D_in * HW_in
    total_out = B * DHW_out
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad3d_kernel[grid](
            x,
            out,
            D_in,
            H_in,
            W_in,
            pad_f,
            pad_t,
            pad_l,
            D_out,
            H_out,
            W_out,
            DHW_out,
            DHW_in,
            HW_out,
            HW_in,
            total_out,
            BLOCK=BLOCK,
        )
    return out.squeeze(0) if is_4d else out


def reflection_pad3d(input: torch.Tensor, padding, *, out: torch.Tensor = None):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD3D")
    return launch_reflection_pad3d(input, padding, out=out)


def reflection_pad3d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD3D_OUT")
    return launch_reflection_pad3d(input, padding, out=out)
