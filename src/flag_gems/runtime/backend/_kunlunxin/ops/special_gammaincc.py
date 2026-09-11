# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of special_gammaincc (aten::special_gammaincc).
#
# Root cause of the generic `flag_gems/ops/special_gammaincc.py` on XPU:
#   The generic wrapper forwards to `flag_gems.ops.igammac_.igammac_func`,
#   a `pointwise_dynamic` kernel whose first act is `lgamma(a_f32)` where
#   `lgamma` is a `@use_tl_extra` stub. `tl_extra_shim` is a shim over the
#   CUDA libdevice, so the symbol is `__nv_lgammaf`, which does not exist on
#   xpu3; the elfconv link step fails with
#       ld.lld: error: undefined symbol: Unsupported
#       >>> referenced by igammac_.py:50
#   i.e. *every* `torch.special.gammaincc` call inside `use_gems()` raised
#   `RuntimeError: Command failed (1): [... xpu3-elfconv-triton ...]`
#   before any kernel ran. `igammac.py` already provides an XPU kernel that
#   evaluates log-gamma inline (Lanczos g=7) and it was already bound to
#   `special_gammaincc.out`, but the functional (out-of-place) aten key was
#   still resolving to the generic implementation, so the whole
#   `-m igammac` marker was red while `-m igammac_out` was green.
#
# Fix: route the functional key at the same vendor kernel that already
# serves the `.out` key. No fallback: this is the XPU Triton kernel from
# `igammac.py`, not an ATen/CPU redispatch.
import logging

import torch

from .igammac import igammac

logger = logging.getLogger(__name__)


def special_gammaincc(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Regularized upper incomplete gamma function Q(a, x) on Kunlunxin XPU."""
    logger.debug("GEMS_KUNLUNXIN SPECIAL_GAMMAINCC")
    return igammac(self, other)
