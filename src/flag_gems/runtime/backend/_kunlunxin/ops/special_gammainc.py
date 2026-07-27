# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of special_gammainc.
#
# Root causes of the generic `flag_gems/ops/special_gammainc.py` on XPU:
#   1. Aten schema `special_gammainc(Tensor, Tensor) -> Tensor` (no out arg),
#      but the generic Python wrapper takes `out=None` by kwarg and never
#      allocates one before forwarding to `_launch_gammainc(out, a, x)`.
#      The dispatcher call `special_gammainc(a, x)` -> `_launch_gammainc(None, ...)`
#      trips `AttributeError: 'NoneType'.device` at line 120 before any kernel.
#   2. The generic series-expansion loop uses a data-dependent early break
#      `if tl.abs(term) < tl.abs(series_sum) * 1e-10: break`. XPU Triton
#      rejects that (`Boolean value of Tensor with more than one value is
#      ambiguous`). The break is only an early-exit optimisation; the terms
#      converge geometrically once i > x, so a fixed 200-iteration loop is
#      mathematically equivalent (the tail terms underflow into f32 noise).
#
# Fix: allocate `out` when None, then run a kernel identical to the generic
# one except the data-dependent break is removed (fixed 200 series iters,
# fixed 300 Lentz continued-fraction iters, matching the generic).
import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def _lgamma_pos(z):
    # Lanczos approximation of log-gamma for z > 0 (g=7, n=9 coefficients).
    # XPU Triton has no `lgamma` intrinsic (undefined symbol at link time),
    # so we evaluate it inline in fp32. gammainc only feeds a > 0, so the
    # reflection formula for z <= 0 is unnecessary here.
    g = 7.0
    x = 0.99999999999980993
    x = x + 676.5203681218851 / (z + 0.0)
    x = x + (-1259.1392167224028) / (z + 1.0)
    x = x + 771.32342877765313 / (z + 2.0)
    x = x + (-176.61502916214059) / (z + 3.0)
    x = x + 12.507343278686905 / (z + 4.0)
    x = x + (-0.13857109526572012) / (z + 5.0)
    x = x + 9.9843695780195716e-6 / (z + 6.0)
    x = x + 1.5056327351493116e-7 / (z + 7.0)
    # Note: coefficients are indexed against (z-1)+i; fold the -1 shift in.
    t = (z - 1.0) + g + 0.5
    half_log_2pi = 0.9189385332046727  # 0.5*log(2*pi)
    return half_log_2pi + ((z - 1.0) + 0.5) * tl.log(t) - t + tl.log(x)


@triton.jit
def gammainc_kernel_xpu(a_ptr, x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    a_f32 = a.to(tl.float32)
    x_f32 = x.to(tl.float32)

    # Edge case default: P(a>0, 0) = 0; NaN otherwise.
    result = tl.where((a_f32 > 0.0) & (x_f32 >= 0.0), 0.0, float("nan"))

    use_series = x_f32 < (a_f32 + 1.0)

    # Series expansion (fixed-count, no data-dep break).
    term = 1.0 / a_f32
    series_sum = term
    for i in range(1, 200):
        term = term * x_f32 / (a_f32 + tl.cast(i, tl.float32))
        series_sum = series_sum + term

    log_gamma_a = _lgamma_pos(a_f32)
    series_result = tl.exp(-x_f32 + a_f32 * tl.log(x_f32) - log_gamma_a) * series_sum

    # Lentz continued fraction for Q(a, x) (large-x regime).
    tiny = 1e-30
    b0 = x_f32 + 1.0 - a_f32
    f_val = b0
    C_val = b0
    D_val = 0.0 * x_f32
    for i_val in range(1, 300):
        i_f = tl.cast(i_val, tl.float32)
        an = i_f * (a_f32 - i_f)
        bn = x_f32 + 2.0 * i_f + 1.0 - a_f32

        D_val = bn + an * D_val
        D_val = tl.where(tl.abs(D_val) < tiny, tiny, D_val)

        C_val = bn + an / C_val
        C_val = tl.where(tl.abs(C_val) < tiny, tiny, C_val)

        D_val = 1.0 / D_val
        delta = C_val * D_val
        f_val = f_val * delta

    log_q = a_f32 * tl.log(x_f32) - x_f32 - log_gamma_a - tl.log(f_val)
    q_val = tl.exp(log_q)
    q_val = tl.where(q_val > 1.0, 1.0, tl.where(q_val < 0.0, 0.0, q_val))
    frac_result = 1.0 - q_val

    result = tl.where(
        (a_f32 > 0.0) & (x_f32 > 0.0),
        tl.where(use_series, series_result, frac_result),
        result,
    )

    tl.store(out_ptr + offsets, result.to(out_ptr.type.element_ty), mask=mask)


def _launch(out, a, x):
    a_c = a.contiguous()
    x_c = x.contiguous()
    was_noncontig = not out.is_contiguous()
    out_c = out.contiguous() if was_noncontig else out
    n = out_c.numel()
    BLOCK = 512
    grid = (triton.cdiv(n, BLOCK),)
    gammainc_kernel_xpu[grid](a_c, x_c, out_c, n, BLOCK_SIZE=BLOCK,
                              buffer_size_limit=2048,
                              isCloseVectorization=True)
    if was_noncontig:
        out.copy_(out_c)
    return out


def special_gammainc(a: torch.Tensor, x: torch.Tensor, *, out: torch.Tensor = None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_GAMMAINC")
    if a.device.type != flag_gems.device:
        raise ValueError(f"gammainc: first input tensor must be on {flag_gems.device}")
    if x.device.type != flag_gems.device:
        raise ValueError(f"gammainc: second input tensor must be on {flag_gems.device}")

    # Broadcast inputs.
    out_shape = torch.broadcast_shapes(a.shape, x.shape)
    if a.shape != out_shape:
        a = a.broadcast_to(out_shape)
    if x.shape != out_shape:
        x = x.broadcast_to(out_shape)

    # Promote non-floating inputs.
    if not a.is_floating_point():
        a = a.to(torch.get_default_dtype())
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())

    if out is None:
        out_dtype = torch.promote_types(a.dtype, x.dtype)
        out = torch.empty(out_shape, dtype=out_dtype, device=a.device)

    # Cast inputs to output dtype (matches generic behaviour).
    if a.dtype != out.dtype:
        a = a.to(out.dtype)
    if x.dtype != out.dtype:
        x = x.to(out.dtype)

    _launch(out, a, x)
    return out
