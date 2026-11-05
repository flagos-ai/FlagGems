import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)


@triton.jit
def _replication_pad1d_kernel_clamp_i64(
    x_ptr,
    out_ptr,
    W_in,
    W_out,
    pad_l,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    o64 = o.to(tl.int64)
    mask = o < total_out

    nc = o64 // W_out
    w_out = o64 % W_out

    iw = w_out - pad_l
    iw = tl.where(iw < 0, 0, iw)
    iw = tl.where(iw > W_in - 1, W_in - 1, iw)

    in_offs = (nc * W_in + iw).to(tl.int64)
    vals = tl.load(x_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o64, vals, mask=mask)


@triton.jit
def _replication_pad1d_kernel_clamp_i32(
    x_ptr,
    out_ptr,
    W_in,
    W_out,
    pad_l,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    nc = o // W_out
    w_out = o % W_out

    iw = w_out - pad_l
    iw = tl.where(iw < 0, 0, iw)
    iw = tl.where(iw > W_in - 1, W_in - 1, iw)

    in_offs = nc * W_in + iw
    vals = tl.load(x_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


@triton.jit
def _replication_pad1d_edge_kernel(
    x_ptr,
    out_ptr,
    W_in,
    W_out,
    pad_l,
    pad_r,
    total_nc,
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    per = pad_l + pad_r
    total_e = total_nc * per
    mask = n < total_e
    nc = n // per
    e = n - nc * per
    is_l = e < pad_l
    dst = nc * W_out + tl.where(is_l, e, pad_l + W_in + (e - pad_l))
    v = tl.load(x_ptr + nc * W_in + tl.where(is_l, 0, W_in - 1), mask=mask)
    tl.store(out_ptr + dst, v, mask=mask)


def _pad2(padding):
    if isinstance(padding, torch.Tensor):
        padding = tuple(int(p) for p in padding.tolist())
    if isinstance(padding, int):
        return (padding, padding)
    if not isinstance(padding, (tuple, list)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of 2 integers: (pad_left, pad_right)"
        )
    return tuple(int(p) for p in padding)


def _launch_flat_clamp(x, out3, W_in, W_out, pad_l, total_out):
    if total_out >= 2**31:
        BLOCK = 1024
        grid = (triton.cdiv(total_out, BLOCK),)
        _replication_pad1d_kernel_clamp_i64[grid](
            x,
            out3,
            W_in,
            W_out,
            pad_l,
            total_out,
            BLOCK=BLOCK,
        )
    else:
        BLOCK = 256 if total_out <= 2048 else 512
        grid = (triton.cdiv(total_out, BLOCK),)
        _replication_pad1d_kernel_clamp_i32[grid](
            x,
            out3,
            W_in,
            W_out,
            pad_l,
            total_out,
            BLOCK=BLOCK,
        )


FLAT_LIMIT = 10_000

_EDGE_BLOCK = 1024


def launch_replication_pad1d(input: torch.Tensor, padding, out: torch.Tensor = None):
    pad_l, pad_r = _pad2(padding)

    dim = input.dim()
    if dim not in (2, 3):
        raise ValueError("replication_pad1d expects 2D (C, W) or 3D (N, C, W) input")

    x = input.contiguous()
    is_2d = dim == 2
    if is_2d:
        x = x.unsqueeze(0)

    N, C, W_in = x.shape
    W_out = W_in + pad_l + pad_r

    if C <= 0:
        raise RuntimeError(
            "Expected 2D or 3D (batch mode) tensor with possibly 0 batch size "
            "and other non-zero dimensions for input"
        )
    if W_in <= 0:
        raise ValueError("Input width must be greater than 0 for replication padding")
    if W_out <= 0:
        raise RuntimeError(
            f"replication_pad1d: output spatial dimension is non-positive: "
            f"output size {W_out}"
        )

    if out is None:
        out3 = torch.empty((N, C, W_out), device=x.device, dtype=x.dtype)
    else:
        expected = (C, W_out) if is_2d else (N, C, W_out)
        if tuple(out.shape) != expected:
            raise ValueError(
                f"Provided out tensor has shape {tuple(out.shape)}, expected {expected}"
            )
        if out.device != x.device:
            raise ValueError("Input and out must be on the same device")
        if out.dtype != x.dtype:
            raise ValueError("Input and out must have the same dtype")
        out3 = out.unsqueeze(0) if is_2d else out

    total_out = N * C * W_out
    if total_out == 0:
        return out3.squeeze(0) if is_2d else out3

    has_neg_pad = pad_l < 0 or pad_r < 0

    if has_neg_pad or total_out <= FLAT_LIMIT:
        kout = out3 if out3.is_contiguous() else torch.empty_like(out3)
        with torch_device_fn.device(x.device):
            _launch_flat_clamp(x, kout, W_in, W_out, pad_l, total_out)
        if kout is not out3:
            with torch_device_fn.device(x.device):
                if not tle_copy(kout, out3):
                    torch.ops.aten._copy_from(kout, out3)
        return out3.squeeze(0) if is_2d else out3

    if not out3.is_contiguous():
        kout3 = torch.empty_like(out3)
        with torch_device_fn.device(x.device):
            dst = torch.narrow(kout3, 2, pad_l, W_in)
            if not tle_copy(x, dst):
                torch.ops.aten._copy_from(x, dst)
            per = pad_l + pad_r
            if per > 0:
                grid = (triton.cdiv(N * C * per, _EDGE_BLOCK),)
                _replication_pad1d_edge_kernel[grid](
                    x,
                    kout3,
                    W_in,
                    W_out,
                    pad_l,
                    pad_r,
                    N * C,
                    BLOCK=_EDGE_BLOCK,
                )
            if not tle_copy(kout3, out3):
                torch.ops.aten._copy_from(kout3, out3)
        return out3.squeeze(0) if is_2d else out3

    with torch_device_fn.device(x.device):
        dst2 = torch.narrow(out3, 2, pad_l, W_in)
        if not tle_copy(x, dst2):
            torch.ops.aten._copy_from(x, dst2)
        per = pad_l + pad_r
        if per > 0:
            grid = (triton.cdiv(N * C * per, _EDGE_BLOCK),)
            _replication_pad1d_edge_kernel[grid](
                x,
                out3,
                W_in,
                W_out,
                pad_l,
                pad_r,
                N * C,
                BLOCK=_EDGE_BLOCK,
            )

    return out3.squeeze(0) if is_2d else out3


def replication_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D")
    return launch_replication_pad1d(input, padding, out=None)


def replication_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D_OUT")
    return launch_replication_pad1d(input, padding, out=out)
