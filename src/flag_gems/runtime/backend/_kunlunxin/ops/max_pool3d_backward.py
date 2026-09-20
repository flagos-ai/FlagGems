import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def max_pool3d_backward_win_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Backward kernel for 3-D max pooling (candidate-window gather).

    Grid: (cdiv(in_D * in_H * in_W, BLOCK), N * C)
    One lane per input position; the output positions whose pooling window may
    contain this element form the small box
    ``[d_lo, d_hi] x [h_lo, h_hi] x [w_lo, w_hi]`` with
    ``o_lo = max(0, ceil((pos + pad - (k-1)*dil) / s))`` and
    ``o_hi = min(out - 1, (pos + pad) // s)``.  For each candidate we load the
    argmax index produced by the forward pass and the upstream gradient, and
    accumulate the gradient whose index equals this input's flat spatial index
    (``d * H * W + h * W + w``).  A candidate whose tap does not land exactly on
    this position can never have this position as argmax, so the index match
    makes the gather exact.

    Design constraints verified on Kunlunxin XPU (2026-08-20/2026-09-11):
    - tl.atomic_add scatter loses updates (~1e-5 per op, seed-dependent) and is
      slow, so every output element's contribution is accumulated in registers
      and written with a single masked store (deterministic, race-free);
    - compound i1-masked loads are a slow path on this backend, so candidate
      addresses are clamped to ``[0, out-1]`` per dim (never out of the (n, c)
      plane) and the loads stay unmasked; the ``o_* <= hi`` gates discard the
      clamped values at value level (same pattern as the forward kernel);
    - runtime trip-count loops crash the backend compiler, so the candidate
      box is bounded by the constexpr ``MAX_*`` (at most 2 per dim for the
      classic k=3/s=2 config) and fully statically unrolled.
    """
    nc = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_hw = in_h * in_w
    in_spatial = in_d * in_hw
    mask = offsets < in_spatial
    safe = tl.where(mask, offsets, 0)
    d = safe // in_hw
    rem2 = safe % in_hw
    h = rem2 // in_w
    w = rem2 % in_w
    my_flat = d * in_hw + h * in_w + w

    d_lo = (d + padding_d - (kernel_d - 1) * dilation_d + stride_d - 1) // stride_d
    d_lo = tl.maximum(d_lo, 0)
    d_hi = (d + padding_d) // stride_d
    d_hi = tl.minimum(d_hi, out_d - 1)
    h_lo = (h + padding_h - (kernel_h - 1) * dilation_h + stride_h - 1) // stride_h
    h_lo = tl.maximum(h_lo, 0)
    h_hi = (h + padding_h) // stride_h
    h_hi = tl.minimum(h_hi, out_h - 1)
    w_lo = (w + padding_w - (kernel_w - 1) * dilation_w + stride_w - 1) // stride_w
    w_lo = tl.maximum(w_lo, 0)
    w_hi = (w + padding_w) // stride_w
    w_hi = tl.minimum(w_hi, out_w - 1)

    out_hw = out_h * out_w
    base = nc.to(tl.int64) * (out_d * out_hw)
    gop = grad_output_ptr + base
    iop = indices_ptr + base
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for od in tl.static_range(0, MAX_D):
        o_d = d_lo + od
        d_ok = o_d <= d_hi
        c_d = tl.minimum(o_d, out_d - 1)
        for oh in tl.static_range(0, MAX_H):
            o_h = h_lo + oh
            h_ok = o_h <= h_hi
            c_h = tl.minimum(o_h, out_h - 1)
            for ow in tl.static_range(0, MAX_W):
                o_w = w_lo + ow
                w_ok = o_w <= w_hi
                c_w = tl.minimum(o_w, out_w - 1)
                o_off = c_d * out_hw + c_h * out_w + c_w
                idx = tl.load(iop + o_off).to(tl.int32)
                val = tl.load(gop + o_off).to(tl.float32)
                ok = mask & d_ok & h_ok & w_ok
                acc += tl.where(ok & (idx == my_flat), val, 0.0)

    tl.store(
        grad_input_ptr + nc.to(tl.int64) * in_spatial + offsets,
        acc,
        mask=mask,
    )


def _parse_pool3d_params(kernel_size, stride, padding, dilation):
    """Parse and validate 3-D pooling parameters.

    Each parameter can be an int (applied to all 3 spatial dims) or a
    3-element tuple/list (D, H, W).
    """

    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return tuple(param)
        raise ValueError(f"Invalid {name}: {param}")

    kd, kh, kw = _parse_param(kernel_size, "kernel_size")
    sd, sh, sw = _parse_param(stride, "stride", default=(kd, kh, kw))
    pd, ph, pw = _parse_param(padding, "padding", default=(0, 0, 0))
    dd, dh, dw = _parse_param(dilation, "dilation", default=(1, 1, 1))

    if sd <= 0 or sh <= 0 or sw <= 0:
        raise ValueError(f"stride must be positive, but got stride=({sd}, {sh}, {sw})")
    if pd < 0 or ph < 0 or pw < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({pd}, {ph}, {pw})"
        )
    if dd <= 0 or dh <= 0 or dw <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dd}, {dh}, {dw})"
        )

    return kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw


def max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    """Backward pass for 3-D max pooling (Kunlunxin)."""
    logger.debug("GEMS_KUNLUNXIN MAX_POOL3D_BACKWARD")
    original_dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32).contiguous()
    indices = indices.to(torch.int32).contiguous()

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = (
        grad_output.shape[2],
        grad_output.shape[3],
        grad_output.shape[4],
    )

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(original_dtype)

    in_spatial = in_d * in_h * in_w
    n_nc = in_n * in_c
    max_d = ((kd - 1) * dd + sd - 1) // sd + 1
    max_h = ((kh - 1) * dh + sh - 1) // sh + 1
    max_w = ((kw - 1) * dw + sw - 1) // sw + 1
    if in_spatial >= 512:
        block = 512
    else:
        block = 1 << (in_spatial - 1).bit_length()
        if block < 32:
            block = 32
    grid = (triton.cdiv(in_spatial, block), n_nc)

    with torch_device_fn.device(grad_input.device):
        max_pool3d_backward_win_kernel[grid](
            grad_output,
            indices,
            grad_input,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            kd,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            dd,
            dh,
            dw,
            max_d,
            max_h,
            max_w,
            block,
            num_warps=4,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return grad_input.to(original_dtype)


def max_pool3d_with_indices_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode: bool,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Backward pass for 3-D max pooling with indices (Kunlunxin).

    Matches the ATen signature of aten::max_pool3d_with_indices_backward, the
    op invoked by the PyTorch autograd formula of max_pool3d_with_indices.
    """
    logger.debug("GEMS_KUNLUNXIN MAX_POOL3D_WITH_INDICES_BACKWARD")
    return max_pool3d_backward(
        grad_output,
        self,
        indices,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
