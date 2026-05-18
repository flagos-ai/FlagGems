import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

atan2 = tl_extra_shim.atan2

logger = logging.getLogger(__name__)

_LOG10E = 0.4342944819032518  # 1 / ln(10)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log10_func(x):
    return tl.log(x.to(tl.float32)) * 0.4342944819032518


# ── complex log10 ─────────────────────────────────────────────────────────────
# log10(a + bi) = log10|z| + i * arg(z) / ln(10)
# where log10|z| = (ln(scale) + 0.5*ln(rn²+in²)) / ln(10)
# and   arg(z)   = atan2(b, a)
# Max-scaling (scale = max(|a|,|b|)) prevents overflow for large magnitudes.


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _log10_complex_real(real, imag):
    LOG10E = 0.4342944819032518
    r = real.to(tl.float32) if real.dtype == tl.float16 else real
    i = imag.to(tl.float32) if imag.dtype == tl.float16 else imag
    abs_r = tl.abs(r)
    abs_i = tl.abs(i)
    scale = tl.where(abs_r > abs_i, abs_r, abs_i)
    safe_scale = tl.where(scale == 0.0, 1.0, scale)
    rn = r / safe_scale
    in_ = i / safe_scale
    log_mag = (tl.log(safe_scale) + 0.5 * tl.log(rn * rn + in_ * in_)) * LOG10E
    return tl.where(scale == 0.0, float("-inf"), log_mag)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _log10_complex_imag(real, imag):
    LOG10E = 0.4342944819032518
    r = real.to(tl.float32) if real.dtype == tl.float16 else real
    i = imag.to(tl.float32) if imag.dtype == tl.float16 else imag
    return atan2(i, r) * LOG10E


def _log10_complex(A):
    """Compute log10 for a complex tensor, returning a complex tensor."""
    real_part = _log10_complex_real(A.real, A.imag)
    imag_part = _log10_complex_imag(A.real, A.imag)
    return torch.complex(real_part, imag_part)


# ── public API ────────────────────────────────────────────────────────────────


def log10(A):
    logger.debug("GEMS LOG10")
    if torch.is_complex(A):
        return _log10_complex(A)
    return log10_func(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    if torch.is_complex(A):
        result = _log10_complex(A)
        A.real.copy_(result.real)
        A.imag.copy_(result.imag)
        return A
    return log10_func(A, out0=A)


def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    if torch.is_complex(A):
        result = _log10_complex(A)
        out.copy_(result)
        return out
    return log10_func(A, out0=out)
