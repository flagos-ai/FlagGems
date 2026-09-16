import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad1d_kernel(
    in_ptr,
    out_ptr,
    W_in,
    pad_left,
    W_out,
    total_out,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)

    b = o // W_out
    w_idx = o - b * W_out

    x = w_idx.to(tl.int32) - pad_left
    pW = 2 * (W_in - 1)
    t = tl.abs(x)
    iw = tl.where(t < W_in, t, pW - t)

    in_offs = b * W_in + iw
    if NEED_MASK:
        mask = o < total_out
        vals = tl.load(in_ptr + in_offs, mask=mask)
        tl.store(out_ptr + o, vals, mask=mask)
    else:
        vals = tl.load(in_ptr + in_offs)
        tl.store(out_ptr + o, vals)


@triton.jit
def copy_tensor_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total
    vals = tl.load(in_ptr + o, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def pad1d_side_kernel(
    in_ptr,
    out_ptr,
    W_in,
    pad_left,
    W_out,
    total_side,
    PAD: tl.constexpr,
    SIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    idx_c = tl.minimum(idx, total_side - 1)
    m = idx < total_side
    b = idx_c // PAD
    j = idx_c - b * PAD
    if SIDE == 0:
        src = pad_left - j
        dst = b * W_out + j
    else:
        src = W_in - 2 - j
        dst = b * W_out + W_in + pad_left + j
    v = tl.load(in_ptr + b * W_in + src)
    tl.store(out_ptr + dst, v, mask=m)


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

    BLOCK = 1024

    if pad_left == 0 and pad_right == 0:
        total = B * W_in
        grid = (triton.cdiv(total, BLOCK),)
        with torch_device_fn.device(x.device):
            copy_tensor_kernel[grid](x, out, total, BLOCK=BLOCK)
        return out

    if W_in < 2:
        raise ValueError(
            "input width must be at least 2 for reflection padding when padding > 0"
        )
    if pad_left >= W_in or pad_right >= W_in:
        raise ValueError(
            "padding values must be less than the input width for reflection padding"
        )

    total_out = B * W_out
    if total_out >= 262144:
        with torch_device_fn.device(x.device):
            mid = torch.ops.aten.slice(out, -1, pad_left, pad_left + W_in)
            if not tle_copy(x, mid):
                torch.ops.aten._copy_from(x, mid, False)
            if pad_left > 0:
                tot = B * pad_left
                pad1d_side_kernel[(triton.cdiv(tot, 1024),)](
                    x,
                    out,
                    W_in,
                    pad_left,
                    W_out,
                    tot,
                    PAD=pad_left,
                    SIDE=0,
                    BLOCK=1024,
                )
            if pad_right > 0:
                tot = B * pad_right
                pad1d_side_kernel[(triton.cdiv(tot, 1024),)](
                    x,
                    out,
                    W_in,
                    pad_left,
                    W_out,
                    tot,
                    PAD=pad_right,
                    SIDE=1,
                    BLOCK=1024,
                )
        return out

    BLOCK = 256 if total_out <= 1024 else 1024
    need_mask = (total_out % BLOCK) != 0
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(x.device):
        reflection_pad1d_kernel[grid](
            x,
            out,
            W_in,
            pad_left,
            W_out,
            total_out,
            BLOCK=BLOCK,
            NEED_MASK=need_mask,
        )
    return out


def reflection_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D")
    return _launch_reflection_pad1d(input, padding, out=None)


def reflection_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D_OUT")
    return _launch_reflection_pad1d(input, padding, out=out)
