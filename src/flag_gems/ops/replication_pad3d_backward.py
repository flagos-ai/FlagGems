import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def replication_pad3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_l,
    pad_t,
    pad_f,
    stride_gon,
    stride_goc,
    stride_god,
    stride_goh,
    stride_gow,
    stride_gin,
    stride_gic,
    stride_gid,
    stride_gih,
    stride_giw,
    C,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_nc = tl.program_id(3)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    c_idx = pid_nc % C
    n_idx = pid_nc // C

    # Output offsets
    od_idx = offs_d
    mask_out = (od_idx < D_out) & (offs_h[:, None, None] < H_out) & (
        offs_w[None, None, :] < W_out
    )

    # Compute corresponding input indices (clamp for replication)
    id_idx = od_idx - pad_f
    id_idx = tl.maximum(id_idx, 0)
    id_idx = tl.minimum(id_idx, D_in - 1)

    ih_idx = offs_h[:, None, None] - pad_t
    ih_idx = tl.maximum(ih_idx, 0)
    ih_idx = tl.minimum(ih_idx, H_in - 1)

    iw_idx = offs_w[None, None, :] - pad_l
    iw_idx = tl.maximum(iw_idx, 0)
    iw_idx = tl.minimum(iw_idx, W_in - 1)

    # Compute base pointers
    go_base = (
        n_idx * stride_gon
        + c_idx * stride_goc
        + od_idx * stride_god
        + offs_h[:, None, None] * stride_goh
        + offs_w[None, None, :] * stride_gow
    )
    gi_base = (
        n_idx * stride_gin
        + c_idx * stride_gic
        + id_idx * stride_gid
        + ih_idx * stride_gih
        + iw_idx * stride_giw
    )

    grad = tl.load(grad_output_ptr + go_base, mask=mask_out, other=0.0)
    grad_f32 = grad.to(tl.float32)

    tl.atomic_add(grad_input_ptr + gi_base, grad_f32, mask=mask_out)


@triton.jit
def _copy_3d_kernel(
    in_ptr, out_ptr, N, C, D, H, W, stride_in, stride_out, BLOCK_W: tl.constexpr
):
    pid = tl.program_id(0)
    n_idx = pid // (C * D * H * (W // BLOCK_W))
    rem = pid % (C * D * H * (W // BLOCK_W))
    c_idx = rem // (D * H * (W // BLOCK_W))
    rem2 = rem % (D * H * (W // BLOCK_W))
    d_idx = rem2 // (H * (W // BLOCK_W))
    h_idx = (rem2 // (W // BLOCK_W)) % H
    pid_w = rem % (W // BLOCK_W)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_w < W) & (n_idx < N) & (c_idx < C) & (d_idx < D) & (h_idx < H)

    in_offset = (
        n_idx * stride_in[0]
        + c_idx * stride_in[1]
        + d_idx * stride_in[2]
        + h_idx * stride_in[3]
        + offs_w * stride_in[4]
    )
    out_offset = (
        n_idx * stride_out[0]
        + c_idx * stride_out[1]
        + d_idx * stride_out[2]
        + h_idx * stride_out[3]
        + offs_w * stride_out[4]
    )

    vals = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + out_offset, vals, mask=mask)


def _launch_replication_pad3d_backward(
    grad_output: torch.Tensor, input: torch.Tensor, padding
):
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = pad_f = pad_ba = padding
    else:
        pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = padding

    grad_output = grad_output.contiguous()
    x = input.contiguous()

    is_4d = x.ndim == 4
    if is_4d:
        x = x.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)

    N, C, D_in, H_in, W_in = x.shape
    D_out, H_out, W_out = (
        D_in + pad_f + pad_ba,
        H_in + pad_t + pad_b,
        W_in + pad_l + pad_r,
    )

    grad_input = torch.zeros_like(x, dtype=torch.float32)

    BLOCK_W = 64
    BLOCK_H = 8
    BLOCK_D = 2

    if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0 and pad_f == 0 and pad_ba == 0:
        # No padding, just copy
        grid = lambda META: (N * C * D_in * H_in * triton.cdiv(W_in, BLOCK_W),)
        with torch_device_fn.device(x.device):
            _copy_3d_kernel[grid](
                grad_output,
                grad_input,
                N,
                C,
                D_in,
                H_in,
                W_in,
                x.stride(),
                grad_input.stride(),
                BLOCK_W=BLOCK_W,
            )
    else:
        grid = lambda META: (
            triton.cdiv(W_out, BLOCK_W),
            triton.cdiv(H_out, BLOCK_H),
            triton.cdiv(D_out, BLOCK_D),
            N * C,
        )
        with torch_device_fn.device(x.device):
            replication_pad3d_backward_kernel[grid](
                grad_output,
                grad_input,
                D_in,
                H_in,
                W_in,
                D_out,
                H_out,
                W_out,
                pad_l,
                pad_t,
                pad_f,
                *grad_output.stride(),
                *x.stride(),
                C,
                BLOCK_D=BLOCK_D,
                BLOCK_H=BLOCK_H,
                BLOCK_W=BLOCK_W,
            )

    # Cast back to original dtype if needed
    if grad_input.dtype == x.dtype:
        result = grad_input
    else:
        result = torch.empty_like(x)
        grid_copy = lambda META: (N * C * D_in * H_in * triton.cdiv(W_in, BLOCK_W),)
        with torch_device_fn.device(x.device):
            _copy_3d_kernel[grid_copy](
                grad_input,
                result,
                N,
                C,
                D_in,
                H_in,
                W_in,
                grad_input.stride(),
                result.stride(),
                BLOCK_W=BLOCK_W,
            )

    return result.squeeze(0) if is_4d else result


def replication_pad3d_backward(grad_output: torch.Tensor, input: torch.Tensor, padding):
    logger.debug("GEMS REPLICATION_PAD3D_BACKWARD")
    return _launch_replication_pad3d_backward(grad_output, input, padding)
