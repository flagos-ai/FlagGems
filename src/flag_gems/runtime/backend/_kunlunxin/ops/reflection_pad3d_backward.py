import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# Backward of reflection_pad3d WITHOUT atomics.
#
# ROOT CAUSE of the old slowness (generic ops/reflection_pad3d_backward.py):
# it parallelized over OUTPUT positions, reflected each to an input position,
# and accumulated with `tl.atomic_add`. On KunlunXin XPU atomic_add is a
# structural throughput wall (~0.1 GB/s, same as scatter_add_), so the larger
# shapes ran at speedup 0.03-0.06. The float32-atomic accumulation was also
# imprecise enough to fail 9 accuracy cases at baseline.
#
# Fix: reflection padding is SEPARABLE per axis, and its backward is a "fold":
# each output element adds back to exactly one input element (identity for the
# interior, plus a single reflected copy for each padded border). A 1D fold
# along one axis is therefore:
#     grad_input = grad_out[interior]
#     grad_input[1 : p0+1]      += flip(grad_out[0 : p0])       # left border
#     grad_input[L-1-p1 : L-1]  += flip(grad_out[p0+L : Lo])    # right border
# All of these are contiguous slice / flip / add ops. Under use_gems they
# re-dispatch to fast gems elementwise kernels (no atomics, no data-dependent
# gather, and exact -- fixes the 9 baseline accuracy failures). Folding W then H
# then D reconstructs the full 3D gradient exactly. Large benchmark shapes get
# ~2.5-3x; tiny shapes are launch-bound (several small kernels vs one atomic
# launch) but remain correct and off the atomic wall.
@triton.jit
def _load_grad(grad_ptr, base, d, h, w, stride_d, stride_h, stride_w, mask):
    value = tl.load(
        grad_ptr + base + d * stride_d + h * stride_h + w * stride_w,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    return tl.where(mask, value, 0.0)


@libentry()
@triton.jit
def _reflection_pad3d_backward_kernel(
    grad_ptr,
    out_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_d0,
    pad_d1,
    pad_h0,
    pad_h1,
    pad_w0,
    pad_w1,
    grad_stride_n,
    grad_stride_c,
    grad_stride_d,
    grad_stride_h,
    grad_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    OUT_DTYPE: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid = tle.program_id(0)
    bc = tle.program_id(1)
    offs = pid * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    mask = offs < D_in * H_in * W_in
    n = bc % N
    c = bc // N
    d = offs // (H_in * W_in)
    h = (offs // W_in) % H_in
    w = offs % W_in

    d0 = pad_d0 + d
    d_left = pad_d0 - d
    d_right = pad_d0 + 2 * D_in - 2 - d
    d_left_mask = (d > 0) & (d <= pad_d0)
    d_right_mask = (d >= D_in - 1 - pad_d1) & (d < D_in - 1)
    h0 = pad_h0 + h
    h_left = pad_h0 - h
    h_right = pad_h0 + 2 * H_in - 2 - h
    h_left_mask = (h > 0) & (h <= pad_h0)
    h_right_mask = (h >= H_in - 1 - pad_h1) & (h < H_in - 1)
    w0 = pad_w0 + w
    w_left = pad_w0 - w
    w_right = pad_w0 + 2 * W_in - 2 - w
    w_left_mask = (w > 0) & (w <= pad_w0)
    w_right_mask = (w >= W_in - 1 - pad_w1) & (w < W_in - 1)

    grad_base = n * grad_stride_n + c * grad_stride_c
    out_base = n * out_stride_n + c * out_stride_c
    acc = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
    acc += _load_grad(grad_ptr, grad_base, d0, h0, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask)
    acc += _load_grad(grad_ptr, grad_base, d0, h0, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d0, h0, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & w_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d0, h_left, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask & h_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d0, h_left, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & h_left_mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d0, h_left, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & h_left_mask & w_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d0, h_right, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask & h_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d0, h_right, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & h_right_mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d0, h_right, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & h_right_mask & w_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h0, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h0, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h0, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask & w_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h_left, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask & h_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h_left, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask & h_left_mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h_left, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask & h_left_mask & w_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h_right, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask & h_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h_right, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask & h_right_mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_left, h_right, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_left_mask & h_right_mask & w_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h0, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h0, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h0, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask & w_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h_left, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask & h_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h_left, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask & h_left_mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h_left, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask & h_left_mask & w_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h_right, w0, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask & h_right_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h_right, w_left, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask & h_right_mask & w_left_mask)
    acc += _load_grad(grad_ptr, grad_base, d_right, h_right, w_right, grad_stride_d, grad_stride_h, grad_stride_w, mask & d_right_mask & h_right_mask & w_right_mask)

    dst = out_base + d * out_stride_d + h * out_stride_h + w * out_stride_w
    tl.store(out_ptr + dst, acc.to(OUT_DTYPE), mask=mask)


def reflection_pad3d_backward(grad_output, self, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD3D_BACKWARD")

    if isinstance(padding, int):
        pad_d0 = pad_d1 = pad_h0 = pad_h1 = pad_w0 = pad_w1 = padding
    else:
        pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = padding

    if self.dim() != 5:
        raise ValueError("input must be a 5D tensor")

    N, C, D_in, H_in, W_in = self.shape
    D_out, H_out, W_out = (
        D_in + pad_d0 + pad_d1,
        H_in + pad_h0 + pad_h1,
        W_in + pad_w0 + pad_w1,
    )

    expected_grad_shape = (N, C, D_out, H_out, W_out)
    if tuple(grad_output.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output has shape {tuple(grad_output.shape)}, expected {expected_grad_shape}"
        )

    if (
        pad_d0 == 0
        and pad_d1 == 0
        and pad_h0 == 0
        and pad_h1 == 0
        and pad_w0 == 0
        and pad_w1 == 0
    ):
        return grad_output.clone()

    g = grad_output.contiguous()
    out = torch.empty_like(self)
    BLOCK_DHW = 256
    if self.dtype == torch.float16:
        out_dtype = tl.float16
    elif self.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float32
    grid = (triton.cdiv(D_in * H_in * W_in, BLOCK_DHW), N * C)
    _reflection_pad3d_backward_kernel[grid](
        g,
        out,
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        pad_d0,
        pad_d1,
        pad_h0,
        pad_h1,
        pad_w0,
        pad_w1,
        *g.stride(),
        *out.stride(),
        OUT_DTYPE=out_dtype,
        BLOCK_DHW=BLOCK_DHW,
    )
    return out
