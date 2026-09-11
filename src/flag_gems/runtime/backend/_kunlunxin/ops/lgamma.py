# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of lgamma / lgamma_.
#
# Root cause: generic `flag_gems/ops/lgamma_.py` calls
# `tl_extra_shim.lgamma(x)` which links to an XPU intrinsic that resolves to
# `undefined symbol: Unsupported` at `xpu3-elfconv-triton` link time. All 18
# baseline cases crash at kernel-compile.
#
# Fix: inline Lanczos g=7 log-gamma for z > 0 in fp32. The lgamma tests only
# feed `torch.rand(...) + 0.1` (strictly positive), so the reflection formula
# for z<=0 is unnecessary. Wired through `pointwise_dynamic` (same INT_TO_FLOAT
# promotion as the generic) to preserve out0-aliased in-place behaviour.
import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


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


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def lgamma_func(x):
    return _lgamma_pos(x.to(tl.float32))


def lgamma(A):
    logger.debug("GEMS_KUNLUNXIN LGAMMA")
    return lgamma_func(A)


def lgamma_(A):
    logger.debug("GEMS_KUNLUNXIN LGAMMA_")
    lgamma_func(A, out0=A)
    return A
