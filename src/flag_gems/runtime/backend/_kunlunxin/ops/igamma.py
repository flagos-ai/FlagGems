# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of igamma (aten::igamma, regularized lower
# incomplete gamma P(a, x), out-of-place).
#
# Root causes of the generic `flag_gems/ops/igamma.py` on XPU:
#   1. Its power-series loop always runs 60 terms, but 24 terms already sit
#      below the fp32 rounding floor for every (a, x) that reaches the series
#      branch (same bound validated for the sibling igammac override; measured
#      wide-domain max error 2.9e-6 with 24 terms once the asymptotic branch
#      covers the large-a / x~a corner). The extra 36 terms are pure cost.
#   2. Its forward modified-Lentz continued fraction needs two divisions, two
#      `tl.abs`, two comparisons and two `tl.where` renormalisation guards per
#      step; the backward recurrence on the same convergent needs one division
#      and four additions and no guards.
#   3. Its block-uniform `if tl.max(x - a) >= 1` keeps the 512-wide series
#      vector live across the CF block, which spills on this backend and
#      roughly doubles the kernel even when the CF body is skipped.
#
# Fix: delegate to the already-tuned sibling kernel `igamma_.igamma_kernel_xpu`
# (24 series terms, backward CF, DLMF 8.12.4 asymptotic branch for a > 10 with
# |sigma| < 0.4, branchless mask-and-sum merge that avoids the uni_sram / spill
# failure of a nested `tl.where`). That kernel is shared with `igamma_`, so the
# math stays identical across the in-place / out-of-place pair.
#
# Historical note: the generic `igamma` and the sibling `igamma_` agree to fp32
# rounding over a wide domain (a in [1e-3, 200], x in [1e-3, 2a], plus the
# a < 0.5 reflection island) once the asymptotic branch is present, and the
# sibling kernel matches an fp64 CPU oracle to <= 3e-6 there.
#
# The only backend-specific launch decision is the tile: on XPU3 the tuned
# kernel is best at BLOCK = 1024 for the large compute-bound shapes, 512 for
# mid tensors, and a narrow 256 tile for tiny (launch-bound) tensors. A fixed
# BLOCK = 1024 regresses the (64, 64) / scalar cases, so the tile is chosen
# from the element count.
import logging

import torch
import triton

import flag_gems

from .igamma_ import igamma_kernel_xpu

logger = logging.getLogger(__name__)

_N_SER = 24
_N_CF = 18


def _block_for(n: int) -> int:
    # Numel-adaptive tile (measured on XPU3, do_bench KERNEL mode). Tiny
    # tensors are launch-bound and prefer one narrow program; large shapes are
    # compute-bound and reach the best throughput at 1024.
    if n <= 1024:
        return 256
    if n <= 8192:
        return 512
    return 1024


def _launch(out: torch.Tensor, a: torch.Tensor, x: torch.Tensor):
    n = out.numel()
    if n == 0:
        return out
    block = _block_for(n)
    grid = (triton.cdiv(n, block),)
    igamma_kernel_xpu[grid](
        a,
        x,
        out,
        n,
        BLOCK_SIZE=block,
        N_SER=_N_SER,
        N_CF=_N_CF,
        buffer_size_limit=2048,
    )
    return out


def igamma(a: torch.Tensor, x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN IGAMMA")
    if not isinstance(a, torch.Tensor):
        raise TypeError("igamma expects a torch.Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("igamma expects a torch.Tensor as the second argument")
    if a.device.type != flag_gems.device:
        raise ValueError(f"igamma: first input tensor must be on {flag_gems.device}")
    if x.device.type != flag_gems.device:
        raise ValueError(f"igamma: second input tensor must be on {flag_gems.device}")
    if a.shape != x.shape:
        raise ValueError("a and x must have the same shape")

    if not a.is_contiguous():
        a = a.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    if a.numel() == 0:
        return torch.empty_like(a)

    out = torch.empty_like(a)
    _launch(out, a, x)
    return out
