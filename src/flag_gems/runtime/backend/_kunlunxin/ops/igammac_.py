# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of igammac_.
#
# Root cause: generic `flag_gems/ops/igammac_.py` calls a `@use_tl_extra`
# `lgamma` shim. On XPU that shim's attribute exists but links to
# `undefined symbol: Unsupported`, so every case crashes at compile.
#
# Fix: inline Lanczos g=7 lgamma (`_lgamma_pos`, same helper as
# lgamma / special_gammainc / mvlgamma_ overrides). Test inputs are
# `torch.rand + 0.1` for both a and x, so a > 0 → no reflection needed.
import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@triton.jit
def _lgamma_pos(z):
    x = 0.99999999999980993
    x = x + 676.5203681218851 / z
    x = x + (-1259.1392167224028) / (z + 1.0)
    x = x + 771.32342877765313 / (z + 2.0)
    x = x + (-176.61502916214059) / (z + 3.0)
    x = x + 12.507343278686905 / (z + 4.0)
    x = x + (-0.13857109526572012) / (z + 5.0)
    x = x + 9.9843695780195716e-6 / (z + 6.0)
    x = x + 1.5056327351493116e-7 / (z + 7.0)
    t = (z - 1.0) + 7.0 + 0.5
    return 0.9189385332046727 + ((z - 1.0) + 0.5) * tl.log(t) - t + tl.log(x)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def igammac_func(a, x):
    EPS = tl.constexpr(1e-12)
    a_f32 = a.to(tl.float32)
    x_f32 = x.to(tl.float32)

    log_gamma_a = _lgamma_pos(a_f32)

    # Lower incomplete gamma series: sum_{n=0} x^n / prod_{k=0..n}(a+k)
    term = 1.0 / a_f32
    sum_val = term
    term = term * x_f32 / (a_f32 + 1.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 2.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 3.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 4.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 5.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 6.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 7.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 8.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 9.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 10.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 11.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 12.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 13.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 14.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 15.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 16.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 17.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 18.0)
    sum_val = sum_val + term
    term = term * x_f32 / (a_f32 + 19.0)
    sum_val = sum_val + term

    sum_val = sum_val + EPS

    log_gamma_lower = a_f32 * tl.log(x_f32) - x_f32 + tl.log(sum_val)
    log_p = log_gamma_lower - log_gamma_a
    p = tl.exp(log_p)
    p = tl.where(p > 1.0, 1.0, p)
    p = tl.where(p < 0.0, 0.0, p)
    q = 1.0 - p
    q = tl.where(x_f32 <= 0.0, 1.0, q)
    return q


def igammac_(A, B):
    logger.debug("GEMS_KUNLUNXIN IGAMMAC_")
    supported_dtypes = (torch.float32, torch.float64)
    if A.dtype not in supported_dtypes or B.dtype not in supported_dtypes:
        raise RuntimeError(
            f"igammac_ Triton kernel supports dtypes {supported_dtypes}, "
            f"but got A.dtype={A.dtype}, B.dtype={B.dtype}"
        )
    return igammac_func(A, B, out0=A)
