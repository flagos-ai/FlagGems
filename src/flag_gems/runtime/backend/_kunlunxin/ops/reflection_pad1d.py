import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad1d_kernel(
    in_ptr, out_ptr, B, W_in, pad_left, W_out, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    # Use modulo wrap to keep all store indices in [0, W_out).
    # On KunlunXin, masked tl.store does not suppress writes for masked-out
    # threads without TRITONXPU_STORE_MASK_SIM=1, causing corruption of
    # adjacent batch row data.  The modulo wrap means tail-block threads simply
    # re-write already-computed values to valid positions — harmless.
    offs_w = (pid_w * BLOCK_W + tl.arange(0, BLOCK_W)) % W_out

    base_in = pid_b * W_in
    base_out = pid_b * W_out

    # Compute reflected indices
    x = offs_w.to(tl.int32) - pad_left  # shift by left pad
    Wm1 = W_in - 1
    p = 2 * Wm1  # period for reflection; guaranteed > 0 when this kernel is used

    t = tl.abs(x)
    m = t % p
    iw = tl.where(m < W_in, m, p - m)

    # No mask needed: offs_w is in [0, W_out) and iw is in [0, W_in)
    vals = tl.load(in_ptr + base_in + iw)
    tl.store(out_ptr + base_out + offs_w, vals)


@triton.jit
def _copy_rows_kernel(in_ptr, out_ptr, B, W, BLOCK_W: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    # Use modulo wrap to avoid masked stores (same KunlunXin workaround).
    offs_w = (pid_w * BLOCK_W + tl.arange(0, BLOCK_W)) % W

    base = pid_b * W
    vals = tl.load(in_ptr + base + offs_w)
    tl.store(out_ptr + base + offs_w, vals)


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
        with torch_device_fn.device(x.device):
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
    with torch_device_fn.device(x.device):
        reflection_pad1d_kernel[grid](x, out, B, W_in, pad_left, W_out, BLOCK_W=256)
    return out


def reflection_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D")
    return _launch_reflection_pad1d(input, padding, out=None)


def reflection_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D_OUT")
    return _launch_reflection_pad1d(input, padding, out=out)
