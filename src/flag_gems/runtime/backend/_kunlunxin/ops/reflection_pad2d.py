import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad2d_kernel(
    in_ptr,
    out_ptr,
    B,
    H_in,
    W_in,
    pad_left,
    pad_top,
    H_out,
    W_out,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Use modulo wrap to keep all store indices in [0, H_out * W_out).
    # On KunlunXin, masked tl.store does not suppress writes for masked-out
    # threads without TRITONXPU_STORE_MASK_SIM=1, causing corruption of
    # adjacent batch row data.  The modulo wrap means tail-block threads simply
    # re-write already-computed values to valid positions — harmless.
    HW_out = H_out * W_out
    offs_n = (pid_n * BLOCK_HW + tl.arange(0, BLOCK_HW)) % HW_out

    # Decode to (h, w) coordinates
    h_idx = offs_n // W_out
    w_idx = offs_n % W_out

    base_in = pid_b * (H_in * W_in)
    base_out = pid_b * HW_out

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

    # No mask needed: offs_n is in [0, HW_out) and in_offs is in [0, H_in*W_in)
    in_offs = ih * W_in + iw
    vals = tl.load(in_ptr + base_in + in_offs)
    tl.store(out_ptr + base_out + offs_n, vals)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, B, H, W, BLOCK_HW: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Use modulo wrap to avoid masked stores (same KunlunXin workaround).
    HW = H * W
    offs_n = (pid_n * BLOCK_HW + tl.arange(0, BLOCK_HW)) % HW

    base = pid_b * HW
    vals = tl.load(in_ptr + base + offs_n)
    tl.store(out_ptr + base + offs_n, vals)


def launch_reflection_pad2d(input: torch.Tensor, padding, out: torch.Tensor = None):
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    # Validate padding values
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be >= 0")

    # Validate input
    if input.dim() < 3:
        raise ValueError("input must have at least 3 dimensions")

    x = input.contiguous()
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    # Validate reflection padding constraints
    if H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 for reflection padding when padding > 0"
        )
    if H_in <= 0 or W_in <= 0:
        raise ValueError("spatial dimensions must be > 0")
    if pad_left >= W_in or pad_right >= W_in or pad_top >= H_in or pad_bottom >= H_in:
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Handle output tensor
    if out is None:
        out = torch.empty(
            (*leading_shape, H_out, W_out), device=x.device, dtype=x.dtype
        )
    else:
        expected_shape = (*leading_shape, H_out, W_out)
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
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        BLOCK_HW = 256
        grid = (B, triton.cdiv(H_in * W_in, BLOCK_HW))
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, B, H_in, W_in, BLOCK_HW=BLOCK_HW)
        return out

    BLOCK_HW = 256
    grid = (B, triton.cdiv(H_out * W_out, BLOCK_HW))
    with torch_device_fn.device(x.device):
        reflection_pad2d_kernel[grid](
            x, out, B, H_in, W_in, pad_left, pad_top, H_out, W_out, BLOCK_HW=BLOCK_HW
        )
    return out


def reflection_pad2d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D")
    return launch_reflection_pad2d(input, padding, out=None)


def reflection_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_OUT")
    return launch_reflection_pad2d(input, padding, out=out)
