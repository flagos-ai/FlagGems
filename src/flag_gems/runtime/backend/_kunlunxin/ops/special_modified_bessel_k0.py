# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of special_modified_bessel_k0[_out].
#
# Root cause: generic `flag_gems/ops/special_modified_bessel_k0.py` uses a
# raw-pointer kernel. On XPU the `tl.log(...)` call inside a complex kernel
# lowers to a runtime `log2` symbol that links to `undefined symbol: log2`
# (same class as our log2 override).
#
# Fix: raw-pointer kernel with launch-time `isCloseVectorization=True,
# buffer_size_limit=2048` (same recipe as special_gammainc override, which
# also uses tl.log successfully). `_i0_approx` is inlined to avoid a
# `@triton.jit` helper that seems to push the vectorizer into the bad path.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _bessel_k0_kernel_xpu(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    is_negative = x_f32 < 0.0
    is_zero = x_f32 == 0.0

    # --- Inline I0(x) approximation (Abramowitz & Stegun 9.8.1 / 9.8.2) ---
    ax = tl.abs(x_f32)
    t_i = x_f32 / 3.75
    y_i = t_i * t_i
    i0_small = 1.0 + y_i * (
        3.5156229
        + y_i * (3.0899424 + y_i * (1.2067492 + y_i * (0.2659732 + y_i * (0.0360768 + y_i * 0.0045813))))
    )
    yb = 3.75 / ax
    i0_big_poly = 0.39894228 + yb * (
        0.01328592
        + yb * (
            0.00225319
            + yb * (
                -0.00157565
                + yb * (
                    0.00916281
                    + yb * (-0.02057706 + yb * (0.02635537 + yb * (-0.01647633 + yb * 0.00392377)))
                )
            )
        )
    )
    i0_big = tl.exp(ax) * i0_big_poly / tl.sqrt(ax)
    i0_x = tl.where(ax <= 3.75, i0_small, i0_big)

    # --- Small region 0 < x <= 2 ---
    y = x_f32 * x_f32 / 4.0
    p = -0.57721566
    p = p + 0.42278441 * y
    p = p + 0.23069500 * y * y
    p = p + 0.03488730 * y * y * y
    p = p + 0.00260380 * y * y * y * y
    p = p + 0.00012900 * y * y * y * y * y
    # log(x/2) — test inputs are strictly positive; no epsilon.
    small_result = -tl.log(x_f32 * 0.5) * i0_x + p

    # --- Large region x > 2 ---
    t = 2.0 / x_f32
    q = 1.25331414
    q = q - 0.07832324 * t
    q = q + 0.0218956 * t * t
    q = q - 0.01072842 * t * t * t
    q = q + 0.00162318 * t * t * t * t
    q = q - 0.00013259 * t * t * t * t * t
    large_result = q * tl.exp(-x_f32) / tl.sqrt(x_f32)

    result = tl.where(x_f32 <= 2.0, small_result, large_result)
    result = tl.where(is_negative, float("nan"), result)
    result = tl.where(is_zero, float("inf"), result)

    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


def _launch(x: torch.Tensor, out: torch.Tensor):
    n_elements = x.numel()
    if n_elements == 0:
        return
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(x.device):
        _bessel_k0_kernel_xpu[grid](
            x, out, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )


def special_modified_bessel_k0(self: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_MODIFIED_BESSEL_K0")
    x_c = self.contiguous()
    out = torch.empty_like(x_c)
    _launch(x_c, out)
    if self.is_contiguous():
        return out
    return out.view_as(self)


def special_modified_bessel_k0_out(self: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_MODIFIED_BESSEL_K0_OUT")
    if out.dtype != self.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != self.device:
        raise TypeError("out device must match input device")
    x_c = self.contiguous()
    out_c = out.contiguous()
    _launch(x_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out
