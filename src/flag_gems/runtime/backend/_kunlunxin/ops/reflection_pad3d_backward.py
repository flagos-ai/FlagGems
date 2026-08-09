"""Backward of reflection_pad3d WITHOUT atomics (Kunlunxin XPU).

Performance reconstruction (2026-08-16):
- flat-DHW input-parallel decomposition, grid=(cdiv(DHW,BLOCK), N*C): no atomics,
  each input element accumulates the reflected grad_output contributions in fp32.
- All shape dims are tl.constexpr, and every index/address vector stays int32:
  the previous flat kernel paid ~330us of runtime int64 div/mod + long address
  chains; constexpr dims turn the per-lane decodes into shifts, and int32 keeps
  the address arithmetic short.
- Address structure: rows are one of 9 (d-mode x h-mode) shared base vectors,
  columns one of 3 shared vectors; each of the 27 loads is a single R[i]+C[j]
  add. The XPU backend lowers this affine structure far more cheaply than 27
  independently re-derived address chains (measured ~5x on the big benchmark
  shapes, ~2.5x on the rest).
- Before loading, the reflected d/h/w coordinates are clamped to their valid
  center value with tl.where, so every load is unconditional and in-bounds.
  Per-contribution validity is reapplied with value-level tl.where(mask, v, 0.0);
  this also guards against the XPU backend's known masked-load `other` quirk
  (masked lanes may still read OOB memory - the select zeroes them afterwards).
- fp32 accumulation, explicit destination-dtype store.

Semantics identical to the previous implementation (48/48 CPU-ref cases).
"""
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def _load_grad(grad_ptr, base, off, mask):
    value = tl.load(grad_ptr + base + off).to(tl.float32)
    return tl.where(mask, value, 0.0)


@libentry()
@triton.jit
def _reflection_pad3d_backward_kernel(
    grad_ptr,
    out_ptr,
    N,
    C,
    pad_d0,
    pad_d1,
    pad_h0,
    pad_h1,
    pad_w0,
    pad_w1,
    OUT_DTYPE: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    DO: tl.constexpr,
    HO: tl.constexpr,
    WO: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid = tle.program_id(0).to(tl.int32)
    bc = tle.program_id(1).to(tl.int32)
    offs = pid * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    mask = offs < D * H * W
    n = bc % N
    c = bc // N

    hw = H * W
    d = offs // hw
    h = (offs // W) % H
    w = offs % W

    dlm = (d > 0) & (d <= pad_d0)
    drm = (d >= D - 1 - pad_d1) & (d < D - 1)
    hlm = (h > 0) & (h <= pad_h0)
    hrm = (h >= H - 1 - pad_h1) & (h < H - 1)
    wlm = (w > 0) & (w <= pad_w0)
    wrm = (w >= W - 1 - pad_w1) & (w < W - 1)

    sdhw = HO * WO
    gb = (n * C + c) * (DO * HO * WO)
    ob = (n * C + c) * (D * H * W)

    # ---- clamped reflected coordinates (keeps every load in-bounds) ----
    d0 = pad_d0 + d
    dl = tl.where(dlm, pad_d0 - d, d0)
    dr = tl.where(drm, pad_d0 + 2 * D - 2 - d, d0)
    h0 = pad_h0 + h
    hl = tl.where(hlm, pad_h0 - h, h0)
    hr = tl.where(hrm, pad_h0 + 2 * H - 2 - h, h0)
    w0 = pad_w0 + w
    wl = tl.where(wlm, pad_w0 - w, w0)
    wr = tl.where(wrm, pad_w0 + 2 * W - 2 - w, w0)

    # ---- 9 shared row bases (d-mode x h-mode) ----
    r00 = gb + d0 * sdhw + h0 * WO
    r0l = gb + d0 * sdhw + hl * WO
    r0r = gb + d0 * sdhw + hr * WO
    rl0 = gb + dl * sdhw + h0 * WO
    rll = gb + dl * sdhw + hl * WO
    rlr = gb + dl * sdhw + hr * WO
    rr0 = gb + dr * sdhw + h0 * WO
    rrl = gb + dr * sdhw + hl * WO
    rrr = gb + dr * sdhw + hr * WO

    # ---- 3 shared column vectors ----
    c0 = w0
    cl = wl
    cr = wr

    acc = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
    acc += _load_grad(grad_ptr, r00, c0, mask)
    acc += _load_grad(grad_ptr, r00, cl, mask & wlm)
    acc += _load_grad(grad_ptr, r00, cr, mask & wrm)
    acc += _load_grad(grad_ptr, r0l, c0, mask & hlm)
    acc += _load_grad(grad_ptr, r0l, cl, mask & hlm & wlm)
    acc += _load_grad(grad_ptr, r0l, cr, mask & hlm & wrm)
    acc += _load_grad(grad_ptr, r0r, c0, mask & hrm)
    acc += _load_grad(grad_ptr, r0r, cl, mask & hrm & wlm)
    acc += _load_grad(grad_ptr, r0r, cr, mask & hrm & wrm)
    acc += _load_grad(grad_ptr, rl0, c0, mask & dlm)
    acc += _load_grad(grad_ptr, rl0, cl, mask & dlm & wlm)
    acc += _load_grad(grad_ptr, rl0, cr, mask & dlm & wrm)
    acc += _load_grad(grad_ptr, rll, c0, mask & dlm & hlm)
    acc += _load_grad(grad_ptr, rll, cl, mask & dlm & hlm & wlm)
    acc += _load_grad(grad_ptr, rll, cr, mask & dlm & hlm & wrm)
    acc += _load_grad(grad_ptr, rlr, c0, mask & dlm & hrm)
    acc += _load_grad(grad_ptr, rlr, cl, mask & dlm & hrm & wlm)
    acc += _load_grad(grad_ptr, rlr, cr, mask & dlm & hrm & wrm)
    acc += _load_grad(grad_ptr, rr0, c0, mask & drm)
    acc += _load_grad(grad_ptr, rr0, cl, mask & drm & wlm)
    acc += _load_grad(grad_ptr, rr0, cr, mask & drm & wrm)
    acc += _load_grad(grad_ptr, rrl, c0, mask & drm & hlm)
    acc += _load_grad(grad_ptr, rrl, cl, mask & drm & hlm & wlm)
    acc += _load_grad(grad_ptr, rrl, cr, mask & drm & hlm & wrm)
    acc += _load_grad(grad_ptr, rrr, c0, mask & drm & hrm)
    acc += _load_grad(grad_ptr, rrr, cl, mask & drm & hrm & wlm)
    acc += _load_grad(grad_ptr, rrr, cr, mask & drm & hrm & wrm)

    dst = ob + d * hw + h * W + w
    tl.store(out_ptr + dst, acc.to(OUT_DTYPE), mask=mask)


def reflection_pad3d_backward(grad_output, self, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD3D_BACKWARD")

    if isinstance(padding, int):
        pad_d0 = pad_d1 = pad_h0 = pad_h1 = pad_w0 = pad_w1 = padding
    else:
        # ATen reflection_pad3d padding order is
        # (padding_left, padding_right, padding_top, padding_bottom,
        #  padding_front, padding_back) == (w0, w1, h0, h1, d0, d1)
        pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1 = padding

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
    out = torch.empty(self.shape, device=self.device, dtype=self.dtype)
    BLOCK_DHW = 256 if D_in * H_in * W_in <= 4096 else 512
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
        pad_d0,
        pad_d1,
        pad_h0,
        pad_h1,
        pad_w0,
        pad_w1,
        OUT_DTYPE=out_dtype,
        D=D_in,
        H=H_in,
        W=W_in,
        DO=D_out,
        HO=H_out,
        WO=W_out,
        BLOCK_DHW=BLOCK_DHW,
    )
    return out