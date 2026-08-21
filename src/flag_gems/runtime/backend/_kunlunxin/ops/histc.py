# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of histc.
#
# Root cause (2026-08-16, XPU 7):
#  - The generic implementation counts via masked `tl.atomic_add`
#    (histc_kernel_simple): masked atomic-add silently drops ~1% of updates
#    on XPU, and masked `other=` loads read real out-of-range memory into
#    the reduction, so many fp32 cases come out wrong.
#  - The previous Kunlunxin override avoided atomics (one program per bin,
#    each streaming the whole input) but still used masked loads with
#    `other=nan`: out-of-bounds lanes are not trustworthy on this backend
#    (torn real reads), so small shapes produced wrong histograms (e.g.
#    returned F on the (64,) test nodes).
#  - tl.histogram is not supported by the XPU backend (PassManager::run
#    failed at make_llir even for a minimal kernel) and unmasked
#    tl.atomic_add is both incorrect (2x-20x overcount) and pathologically
#    slow (~14M atomics/s) on this backend, so a single-pass histogram
#    cannot be built from Triton primitives here.
#
# Fix: deterministic per-bin counting with no masked memory anywhere.
#  Every lane lands on a clamped in-bounds offset; out-of-range / NaN /
#  OOB lanes are zeroed by an integer ok-multiplier (the count_nonzero
#  clamp+mult pattern).  bins <= 100 in the test/benchmark matrix so the
#  extra passes over the data are cheap and, crucially, exact.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def histc_bin_main_kernel(
    inp_ptr,
    out_ptr,
    n_main,
    bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per bin over the contiguous part only (n_main is a
    # multiple of BLOCK_SIZE): pure contiguous loads, no masks, no clamped
    # (gather) addresses, so full streaming bandwidth is kept.
    b = ext.program_id(0)
    inv_scale = bins / (max_val - min_val)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for start in range(0, n_main, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        v = tl.load(inp_ptr + offs).to(tl.float32)
        idx = tl.floor((v - min_val) * inv_scale).to(tl.int32)
        # elements exactly at max_val belong to the last bin
        idx = tl.where(v == max_val, bins - 1, idx)
        in_range = ((v >= min_val) & (v <= max_val)).to(tl.int32)
        hit = (idx == b).to(tl.int32) * in_range
        acc += hit
    total = tl.sum(acc)
    tl.store(out_ptr + b, total.to(tl.float32))


@libentry()
@triton.jit
def histc_bin_tail_kernel(
    inp_ptr,
    out_ptr,
    n_tail,
    off,
    bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Tail (< one BLOCK_SIZE) with clamped offsets + ok-multiplier
    # (masked `other=` values are not reliable on this backend).  Only the
    # tail tile ever gathers, so the main pass keeps contiguous streaming.
    b = ext.program_id(0)
    inv_scale = bins / (max_val - min_val)
    offs = off + tl.arange(0, BLOCK_SIZE)
    last = off + n_tail - 1
    cclamp = tl.minimum(offs, last)
    ok = (offs <= last).to(tl.int32)
    v = tl.load(inp_ptr + cclamp).to(tl.float32)
    idx = tl.floor((v - min_val) * inv_scale).to(tl.int32)
    idx = tl.where(v == max_val, bins - 1, idx)
    in_range = ((v >= min_val) & (v <= max_val)).to(tl.int32)
    hit = (idx == b).to(tl.int32) * in_range * ok
    total = tl.sum(hit)
    # add to the main pass result (single program per bin, no race)
    prev = tl.load(out_ptr + b)
    tl.store(out_ptr + b, total.to(tl.float32) + prev)


@libentry()
@triton.jit
def histc_range_kernel(
    inp_ptr,
    mn_ptr,
    mx_ptr,
    n,
    TILE: tl.constexpr,
    GRID: tl.constexpr,
):
    # mask-free grid-stride min/max; clamped lanes duplicate the last
    # element, which does not change the min/max. Avoids the gem min/max
    # path whose 2D variant wedges this device (66250 kernel exception).
    pid = ext.program_id(0)
    last = n - 1
    mmin = tl.full((TILE,), float("inf"), dtype=tl.float32)
    mmax = tl.full((TILE,), float("-inf"), dtype=tl.float32)
    for start in range(pid * TILE, n, GRID * TILE):
        cols = start + tl.arange(0, TILE)
        cclamp = tl.minimum(cols, last)
        v = tl.load(inp_ptr + cclamp).to(tl.float32)
        mmin = tl.minimum(mmin, v)
        mmax = tl.maximum(mmax, v)
    tl.store(mn_ptr + pid, tl.min(mmin, axis=0))
    tl.store(mx_ptr + pid, tl.max(mmax, axis=0))


@libentry()
@triton.jit
def kern_range_combine(
    mn_ptr,
    mx_ptr,
    out_mn,
    out_mx,
    G: tl.constexpr,
    NG: tl.constexpr,
):
    idx = tl.arange(0, G)
    cclamp = tl.minimum(idx, NG - 1)
    mn = tl.min(tl.load(mn_ptr + cclamp), axis=0)
    mx = tl.max(tl.load(mx_ptr + cclamp), axis=0)
    tl.store(out_mn, mn)
    tl.store(out_mx, mx)


def _data_range(inp):
    n = inp.numel()
    grid = 1
    tiles = triton.cdiv(n, 8192)
    want = triton.cdiv(tiles, 8)
    while grid < 256 and grid * 2 <= want:
        grid *= 2
    mn_part = torch.empty(grid, dtype=torch.float32, device=inp.device)
    mx_part = torch.empty(grid, dtype=torch.float32, device=inp.device)
    out_mn = torch.empty((), dtype=torch.float32, device=inp.device)
    out_mx = torch.empty((), dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        histc_range_kernel[(grid,)](
            inp, mn_part, mx_part, n,
            TILE=8192, GRID=grid,
        )
        g_pow = 1
        while g_pow < grid:
            g_pow *= 2
        kern_range_combine[(1,)](
            mn_part, mx_part, out_mn, out_mx, G=g_pow, NG=grid,
        )
    return float(out_mn.item()), float(out_mx.item())


def histc(inp, bins=100, min=0, max=0):
    logger.debug("GEMS_KUNLUNXIN HISTC")

    inp = inp.contiguous()

    min_val = float(min)
    max_val = float(max)

    if min_val == 0 and max_val == 0:
        min_val, max_val = _data_range(inp)

    if min_val == max_val:
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        # torch's CPU reference places all-equal data at bin = bins // 2
        # (write via the native strided-copy engine; the gem copy_ does not
        # support scalar-element assignment on Kunlunxin)
        ones = torch.full((1,), count, dtype=inp.dtype, device=inp.device)
        torch.ops.aten._copy_from(ones, out[bins // 2 : bins // 2 + 1], False)
        return out

    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (bins,)

    n_main = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    n_tail = n_elements - n_main

    with torch_device_fn.device(inp.device):
        if n_main:
            histc_bin_main_kernel[grid](
                inp,
                out,
                n_main,
                bins,
                min_val,
                max_val,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        if n_tail:
            histc_bin_tail_kernel[grid](
                inp,
                out,
                n_tail,
                n_main,
                bins,
                min_val,
                max_val,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out