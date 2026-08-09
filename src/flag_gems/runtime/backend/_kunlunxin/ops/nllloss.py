# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sum import sum as xpu_sum
from .zero import zero_
from .zeros_like import zeros_like as xpu_zeros_like

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
    PADDED: tl.constexpr = False,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_n

    if wgt_ptr is None:
        # `ignore_mask.to(tl.float32)` (arith.uitofp on a bool tile) makes
        # TritonXPUUnrollControl fail to verify for BLOCK_N >= 256 in fp16;
        # the explicit select is equivalent and compiles for every block.
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    if PADDED:
        # Scratch buffers are over-allocated to a whole number of reduction
        # tiles; lanes >= N already carry 0 (ignore_mask is false there), so an
        # unmasked store zero-fills the pad and lets the reduction kernel run
        # on fully unmasked tiles.
        tl.store(out_ptr + offsets_n, out)
        tl.store(ignore_wgt_tgt_ptr + offsets_n, wgt_tgt)
    else:
        tl.store(out_ptr + offsets_n, out, mask=mask_n)
        if reduction != 0:
            tl.store(ignore_wgt_tgt_ptr + offsets_n, wgt_tgt, mask=mask_n)


# ---------------------------------------------------------------------------
# Fused scalar reduction for the reduced (mean/sum) nll_loss forward paths.
#
# Baseline used `xpu_sum(out)` + `xpu_sum(ignore_weight_tgt)` + a gems 0-d
# `div` (3 to 7 device launches, and the 0-d pointwise `div` alone measured
# ~0.10-0.14 ms). This single-program kernel replaces all of them with one
# launch: it reduces both scratch buffers and writes the final scalar loss
# plus total_weight.
#
# XPU reduction constraints honored here (harness/HARNESS_SUMMARY.md 2.5/3.6):
#   - every reduction tile is <= 8192 lanes and *fully unmasked*: the caller
#     over-allocates the scratch buffers to `NTILES * TL` elements and the
#     elementwise kernel zero-fills the pad (`PADDED=True`).  A masked tail
#     tile inside this kernel was measured to silently miscompile (70/216
#     probe cases wrong at N=10000), matching the documented restriction.
#   - no `tl.where(...)` inside the reduction, no atomics, no runtime-loop
#     masked loads (`tl.static_range` only).
#   - `NTILES` is capped by the caller so the unrolled body and the live
#     local-memory footprint stay small; larger N keeps the staged
#     `xpu_sum` path.
# ---------------------------------------------------------------------------
_NLL_REDUCE_TILE = 8192
_NLL_REDUCE_MAX_TILES = 2
# Elementwise block for the 1d/2d forward gather.  Measured kernel-only medians
# on XPU (fp32, do_bench): BLOCK 128/256/512 give 0.0090/0.0085/0.0116 ms at
# N=64, 0.0229/0.0165/0.0123 ms at N=4096 and 0.0389/0.0289/0.0241 ms at
# N=10000 - wide blocks (fewer, fatter programs) win once N passes ~1K while
# the extra masked lanes dominate below it, hence the two-point split.
# NOTE: blocks >= 256 only compile after replacing `ignore_mask.to(tl.float32)`
# with an explicit `tl.where` in the kernel (see the comment there).
_NLL_FWD_BLOCK_SMALL = 256
_NLL_FWD_BLOCK_LARGE = 512
_NLL_FWD_BLOCK_SWITCH = 1024


def _nll_fwd_block(n):
    return _NLL_FWD_BLOCK_SMALL if n <= _NLL_FWD_BLOCK_SWITCH else _NLL_FWD_BLOCK_LARGE


@libentry()
@triton.jit
def nll_loss_reduce_kernel(
    out_ptr,
    wgt_ptr,
    total_out_ptr,
    total_wgt_ptr,
    MEAN: tl.constexpr,
    NTILES: tl.constexpr,
    TL: tl.constexpr,
):
    total_o = tl.zeros([], dtype=tl.float32)
    total_wgt = tl.zeros([], dtype=tl.float32)
    for i in tl.static_range(NTILES):
        off = i * TL + tl.arange(0, TL)
        o = tl.load(out_ptr + off).to(tl.float32)
        w = tl.load(wgt_ptr + off).to(tl.float32)
        total_o += tl.sum(o)
        total_wgt += tl.sum(w)

    if MEAN:
        res = total_o / total_wgt
    else:
        res = total_o
    tl.store(total_out_ptr, res.to(total_out_ptr.dtype.element_ty))
    tl.store(total_wgt_ptr, total_wgt.to(total_wgt_ptr.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offsets_n
        out_grad = tl.load(out_grad_ptrs, mask=mask_n, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)
    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=ignore_mask)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    out_ptrs = out_ptr + offset_n * D + offset_d
    tl.store(out_ptrs, out, mask=mask_block)

    if reduction == 1:
        ignore_wgt_tgt_ptrs = ignore_wgt_tgt_ptr + offset_n * D + offset_d
        tl.store(ignore_wgt_tgt_ptrs, wgt_tgt, mask=mask_block)


# ---------------------------------------------------------------------------
# Row-tiled variant of the 2d forward gather.  Used by `nll_loss2d_forward`
# only (grep-verified single call site); the flat kernel above stays as the
# general fallback for shapes this one cannot serve safely.
#
# The flat kernel spends most of its time *outside* the (unavoidable) discrete
# gather: with `BLOCK_ND=128` over `N*D` it launches `N*D/128` tiny programs and
# recomputes `offset_nd % D` / `offset_nd // D` per element, which also stops
# OffsetAnalysis from proving the target/out accesses stride-1.  Here the row
# index is `program_id(1)` (a scalar) and `D` is `tl.constexpr`, so target/out
# become contiguous block DMA and the div/mod disappear.  Measured on
# `(64, 512, 512)` (the only benchmark shape that reaches this op), wrapper
# level medians, reduction=none: fp16 0.2356 -> 0.1310, fp32 0.3096 -> 0.1035,
# bf16 0.2591 -> 0.1188 ms.
#
# XPU constraints honored (harness/HARNESS_SUMMARY.md 2.3/2.4/3.6):
#   - the tile is *exactly* `TILE_N x BLOCK_D` with `D % BLOCK_D == 0`, so
#     there is no masked tail.  A masked tail silently miscompiles here:
#     `BLOCK_D=1024` over `D=512` reported success while producing fp32
#     `maxdiff=6.918`.
#   - `BLOCK_D >= _NLL2D_MIN_BLOCK_D` (128).  Narrow inner tiles do not just
#     return wrong values on this backend, they fault the device: `BLOCK_D=32`
#     (with `TILE_N=2`) and `BLOCK_D=64` (with `TILE_N=8`) both raised
#     `kl3ChannelCheckErrors error 721` -> `wait for noc idle timeout` and
#     wedged the card (reproduced with the in-kernel `assert` removed, so the
#     trigger is the tile width, not the assert).  128/256/512 verified clean.
#   - `TILE_N` stays 1: `TILE_N` 2/4/8/16 at `BLOCK_D=512` are correct but
#     slower (0.0737/0.0899/0.0911/0.1186 vs 0.0692 ms).
#   - the `wgt_ptr is None` branch uses `tl.where` instead of
#     `ignore_mask.to(tl.float32)`: the uitofp form makes
#     `TritonXPUUnrollControl` fail (`OutOfResources: uni_sram`) for fp16 at
#     `BLOCK_D=512`.  This `tl.where` is elementwise, not inside a reduction.
# ---------------------------------------------------------------------------
_NLL2D_MIN_BLOCK_D = 128
_NLL2D_MAX_BLOCK_D = 512


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_tiled_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    C,
    reduction: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_N: tl.constexpr = 1,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_n * TILE_N + tl.arange(0, TILE_N))[:, None]
    cols = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    flat = rows * D + cols

    tgt = tl.load(tgt_ptr + flat)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = tgt != ignore_index

    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + rows * (C * D) + tgt * D + cols
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    tl.store(out_ptr + flat, out)
    if reduction == 1:
        tl.store(ignore_wgt_tgt_ptr + flat, wgt_tgt)


# ---------------------------------------------------------------------------
# Wide flat variant of the 2d forward gather, for trailing dims too small for
# `nll_loss2d_forward_tiled_kernel` (`D < 128`, e.g. the `(*, *, 4, 8)` shapes
# in the benchmark matrix, where `D == 32`).
#
# The original flat kernel above launches `ceil(N*D / 128)` programs.  On this
# backend the per-program cost is set by the tile's *byte* width, not by the
# live data (renorm_ calibration: 256 B -> ~0.83 us/program, >= 1024 B -> ~0.11
# us/program), so `BLOCK_ND=128` (256 B in fp16) is far too narrow: `N*D=131072`
# (`(4096, 4096, 4, 8)`) means 1024 programs, and the measured per-cell minimum
# for that cell's `reduction=none` really is 373-503 us against a 29-35 us torch
# reference.
#
# Differences from the flat kernel above, all of them safe rewrites:
#   - `D` is `tl.constexpr`, so `offset_nd % D` / `offset_nd // D` become a
#     mask/shift instead of runtime integer div/mod.
#   - `target`/`out`/`ignore_weight_tgt` are addressed with the flat index
#     directly: `target_flat`/`out` are contiguous `(N, D)`, so
#     `offset_n * D + offset_d == offset_nd` identically.  This turns three
#     accesses that OffsetAnalysis could not prove stride-1 into contiguous
#     block DMA.
#   - the caller only selects this kernel when `BLOCK_ND` divides `N*D`
#     exactly, so the kernel is *completely unmasked* on those three accesses:
#     no `other=` (documented as not honoured on this backend even for
#     stride-1 loads) and no masked tail store (documented to write the whole
#     tile, i.e. up to `BLOCK_ND-1` elements past the end).  Shapes that do not
#     divide keep the original 128-wide masked kernel unchanged.
#   - the `wgt_ptr is None` branch uses `tl.where` rather than
#     `ignore_mask.to(tl.float32)`; the uitofp form fails
#     `TritonXPUUnrollControl` for BLOCK >= 256 in fp16.
# The discrete `inp` gather keeps HEAD's exact expression and masking so the
# (pre-existing) out-of-range `tgt*D` read behaviour for a hit `ignore_index`
# is bit-for-bit unchanged.
# ---------------------------------------------------------------------------
# `BLOCK_ND = 1024` is **silently wrong** in this kernel: with fp32 input and a
# `weight` tensor it returns an exact `0.0` for ~5-6% of the lanes (isolated
# probe, CPU float64 oracle, mismatch == exact-zero count):
#     M=2048   96/2048     M=4096  218/4096
#     M=8192  456/8192     M=16384 982/16384
# while 128 / 256 / 512 / 2048 all give 0 mismatches on the same data, and fp16
# / bf16 are unaffected at every width.  The defect is therefore *width*
# specific and non-monotone (the documented TritonXPU pattern), so the width is
# taken from an explicitly verified whitelist rather than from `next_pow2`.
_NLL2D_FLAT_BLOCKS = (2048, 512, 256, 128)


def _nll2d_flat_cap(M):
    """Measured best `BLOCK_ND` band for the wide flat kernel.

    Wrapper-level `do_bench` minima over 3 interleaved rounds, `weight=None`
    (the weight-gather cells are bimodal and unusable for tuning), block cap
    swept 0(=HEAD)/128/256/512/1024/2048/4096/8192, fp16 / fp32 in us:

      M=2048    39.6/23.9  20.7/15.2  13.9/11.7  *12.6/11.3*  14.3/14.5  20.6/21.1  ...
      M=8192    68.2/55.0  35.6/29.7  26.0/20.5   19.2/15.1   [14.9/14.8] 21.8/21.6  35.5/35.4  61.9/61.3
      M=131072  425/375    223/194    150/135     113/98       96.8/90.3 *89.9/88.8* 94.9/94.4  117/116

    The optimum grows with `M` (more parallelism needed) but a single tile that
    covers all of `M` is always bad (M=8192 @ 8192 -> 61 us vs 14.9 us @ 1024).
    The M=8192 optimum (1024, in brackets) is *not* usable - see
    `_NLL2D_FLAT_BLOCKS` - so 512 is taken there instead.
    """
    return 512 if M <= 65536 else 2048


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_flat_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    C,
    reduction: tl.constexpr,
    D: tl.constexpr,
    BLOCK_ND: tl.constexpr,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)

    tgt = tl.load(tgt_ptr + offset_nd)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = tgt != ignore_index

    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    offset_d = offset_nd % D
    offset_n = offset_nd // D
    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    tl.store(out_ptr + offset_nd, out)
    if reduction == 1:
        tl.store(ignore_wgt_tgt_ptr + offset_nd, wgt_tgt)


def _nll2d_flat_block(M):
    """Widest verified-safe block that divides `M` and fits `_nll2d_flat_cap(M)`.

    `None` means "no unmasked wide tiling possible"; the caller then keeps the
    original 128-wide masked flat kernel.
    """
    cap = _nll2d_flat_cap(M)
    for block in _NLL2D_FLAT_BLOCKS:
        if block <= cap and M % block == 0:
            return block
    return None


def _nll2d_block_d(D):
    """Largest power-of-two `BLOCK_D` in [128, 512] that exactly divides `D`.

    Returns `None` when no such value exists (odd / small trailing dims), in
    which case `nll_loss2d_forward` keeps the original flat kernel.
    """
    block_d = 1
    cap = min(D, _NLL2D_MAX_BLOCK_D)
    while block_d * 2 <= cap and D % (block_d * 2) == 0:
        block_d *= 2
    return block_d if block_d >= _NLL2D_MIN_BLOCK_D else None


# ---------------------------------------------------------------------------
# Fused scalar reduction for the *2d/nd* reduced (mean/sum) forward paths.
#
# Same idea (and the same kernel) as the 1d path above.  HEAD finishes the 2d
# reduced paths on the host with `xpu_sum(out)` [+ `xpu_sum(ignore_weight_tgt)`
# + a 0-d gems `div`] + `.to(dtype)`, i.e. 3-7 extra device launches.  Measured
# per-cell minima over 4 benchmark rounds on XPU7 (`benchmark/test_nll_loss_nd.py
# -m nll_loss_nd_forward`, gems latency in us) show that tail dominates every
# reduced cell:
#
#   shape                 reduction=none   sum          mean
#   (64, 64, 4, 8)        24 - 41          49 - 58      155 - 162
#   (256, 256, 4, 8)      55 - 86          58 - 90      153 - 175
#   (32, 128, 512)        18 - 86          82 - 181     220 - 273
#   (64, 64, 8, 4, 8)     30 - 87          82 -  92     220 - 242
#
# i.e. `mean - none` is a flat +140..+170 us (the 0-d gems `div` alone is
# documented at 105-139 us) and `sum - none` is +25..+65 us, while the torch
# reference sits at 11-35 us for *all* of them.
#
# `nll_loss_reduce_kernel` needs fully unmasked reduction tiles, so the fused
# path is only taken when the `N*D` element count is *exactly* a whole number
# of tiles.  That is deliberately stricter than padding the scratch buffers:
# padding would need either an extra memset launch (eating the win) or a new
# unmasked-store mode in both 2d gather kernels.  Every 3d+ shape in the
# official benchmark matrix has a power-of-two `N*D` (2048 / 8192 / 16384 /
# 131072) so nothing is lost there, and non-conforming shapes keep the exact
# HEAD `xpu_sum` path.
#
# Restricted to `reduction=mean` on purpose.  Two `sum` variants were measured
# and both rejected:
#   - making the gather kernel also write the per-element weight scratch so the
#     weight sum would be real: (64,64,4,8) sum 49-58 -> 61-71 us,
#     (256,256,4,8) sum 58-90 -> 78-133 us (the extra store costs more than the
#     `xpu_sum` it saves);
#   - reusing this two-buffer kernel with `out` handed in twice (no extra
#     store): net 1.032x over the 30 owned `sum` cells - it wins on the tiled
#     shapes (M=16384: 78-82 -> 51-57 us, 1.42-1.57x) but loses on the wide-flat
#     shapes (M=2048/8192: 35-43 -> 45-52 us, 0.73-0.91x) because `xpu_sum` is
#     multi-program while this reduce kernel is a single program.  Whole-matrix
#     per-cell-MIN dtype-equal-weight came out at 0.5794 with it vs 0.5861
#     without, so it was reverted.
# `sum` therefore keeps HEAD's path byte-for-byte.
#
# `_NLL2D_REDUCE_MAX_TILES` must stay at 2.  Raising it to 16 so that the
# largest benchmark cell (`N*D = 131072`) could fuse too **faults the device**:
# fp32 `(4096, 64, 4, 8)` mean raised `kl3ChannelCheckErrors ... status=719`,
# `cluster[11] ... reason[26] sm rdwr conflict` and
# `Xid (PCI:0000:da:00): KL_XID_KERNEL_EXCEPTION` inside
# `nll_loss_reduce_kernel` (dmesg kl3_dev7, 2026-08-30; the same config is
# numerically correct in fp16).  Do not retry.
# ---------------------------------------------------------------------------
_NLL2D_REDUCE_TILE = 8192
_NLL2D_REDUCE_MAX_TILES = 2


def _nll2d_fused_reduce(M):
    """Return `(ntiles, TL)` if `M` elements can be reduced by one unmasked
    fused launch, else `None`."""
    tl_width = min(_NLL2D_REDUCE_TILE, triton.next_power_of_2(M))
    ntiles = triton.cdiv(M, tl_width)
    if ntiles > _NLL2D_REDUCE_MAX_TILES or ntiles * tl_width != M:
        return None
    return ntiles, tl_width


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offset_n * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=mask_block, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1
    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offset_n * C * D + tgt * D + offset_d
    tl.store(inp_grad_ptrs, inp_grad, mask=ignore_mask)


# ---------------------------------------------------------------------------
# Wide flat variant of the 2d/nd backward scatter.
#
# `nll_loss2d_backward_kernel` above runs `ceil(N*D / 128)` programs.  A
# component breakdown of the official matrix (wrapper-level `do_bench` minima,
# XPU 7, us) shows the scatter kernel - not the zero fill - is what the small
# and medium shapes spend their time on, and that the reduced paths cost about
# twice the `reduction=none` path:
#
#   shape              N*D      zero_   kern(none)  kern(mean)
#   (64, 64, 4, 8)     2048       5.0        9.5        11.3
#   (256, 256, 4, 8)   8192       8.9       17.7        29.5
#   (32, 128, 512)     16384      8.9       29.9        55.1
#   (4096, 4096, 4, 8) 131072  1203.3      184.5       398.0
#
# The `mean`/`sum` surcharge is exactly the documented cost of loading a scalar
# from global memory inside the kernel (0.21-0.27 us per program per load):
# `N*D = 131072` at `BLOCK_ND = 128` is 1024 programs and the surcharge is
# 398 - 184 = 214 us ~= 1024 * 0.21 us.  Both that surcharge and the per-program
# tile overhead shrink linearly with the tile width, so the fix is to widen the
# tile; the same measurement also explains why widening helps `reduction=none`
# much less (1.5x instead of 2.5x).
#
# Differences from the kernel above, all of them provably equivalent rewrites:
#   - `D` is `tl.constexpr`, so `offset_nd % D` / `offset_nd // D` are folded by
#     the compiler instead of being runtime integer div/mod.
#   - `target` and (for `reduction=none`) `grad_output` are addressed with the
#     flat index directly: both are contiguous `(N, D)`, so
#     `offset_n * D + offset_d == offset_nd` identically.  That turns two
#     accesses OffsetAnalysis could not prove stride-1 into contiguous block
#     DMA.
#   - the caller only selects this kernel when `BLOCK_ND` divides `N*D`
#     exactly, so both of those loads are *completely unmasked*: no `other=`
#     (documented as not honoured on this backend even for stride-1 loads) and
#     no masked tail.  Shapes that do not divide keep the original 128-wide
#     masked kernel unchanged.
#   - the scatter target is clamped with `safe_tgt` and the store is unmasked.
#     This is not a relaxation, it is a fix: masked stores are documented not to
#     be honoured on this backend, and HEAD's store address for an ignored lane
#     is `offset_n * C * D + ignore_index * D + offset_d`, which for the default
#     `ignore_index = -100` is *before* the buffer and for `ignore_index = 200`
#     with `C = 64` runs past its end.  Because every `(n, d)` lane owns a
#     unique output address, writing an explicit `0.0` to `(n, 0, d)` for an
#     ignored lane is exactly the value `grad_input` already holds from the
#     zero fill, so the result is unchanged while the write stays in bounds.
#   - the weight gather uses `safe_tgt` too, which removes the equally
#     out-of-range `wgt_ptr + ignore_index` read.  Measured cost-neutral: three
#     variants (HEAD's `mask=ignore_mask, other=0` on the raw `tgt`, unmasked on
#     `safe_tgt`, masked on `safe_tgt`) differ by <=2% over 24 configurations and
#     all three give 0 mismatches against a CPU float64 oracle.
#
# Width whitelist: 128/256/512/1024/2048/4096/8192 were all swept against a CPU
# float64 oracle over 5 shapes x 3 dtypes x weight{Y,N} x reduction{0,1,2}
# (441 checks) with **0 mismatches everywhere**, i.e. this kernel does not show
# the width-specific silent miscompile that `nll_loss2d_forward_flat_kernel`
# has at 1024.  The whitelist below is therefore a *performance* band, and the
# widths in it are exactly the ones that were verified.
# ---------------------------------------------------------------------------
_NLL2D_BWD_FLAT_BLOCKS = (4096, 1024, 512, 256, 128)


def _nll2d_bwd_flat_cap(M):
    """Measured best `BLOCK_ND` band for the wide flat backward scatter.

    Kernel-only `do_bench` minima, summed over 3 dtypes x weight{Y,N} x
    reduction{none,mean,sum} (18 cells per row, us):

      N*D      HEAD(128)  128    256    512    1024   2048   4096   8192
      2048        255     240    220    220    ---    ---     -      -
      8192        515     ---    ---    385    370    ---     -      -
      16384      1134     ---    ---    ---    900    925     -      -
      131072     9383     ---    ---    ---    ---   7765   7577   8000+

    A single tile covering all of `N*D` is always bad, and the optimum grows
    with `N*D`; the weight-gather cells are flat in the width (the discrete
    gather dominates them) so the band is set by the `weight=None` cells.
    """
    if M <= 2048:
        return 512
    if M <= 65536:
        return 1024
    return 4096


def _nll2d_bwd_flat_block(M):
    """Widest whitelisted block that divides `M` and fits `_nll2d_bwd_flat_cap`.

    `None` means "no unmasked wide tiling possible"; the caller then keeps the
    original 128-wide masked flat kernel byte-for-byte.
    """
    cap = _nll2d_bwd_flat_cap(M)
    for block in _NLL2D_BWD_FLAT_BLOCKS:
        if block <= cap and M % block == 0:
            return block
    return None


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_flat_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    C,
    reduction: tl.constexpr,
    D: tl.constexpr,
    BLOCK_ND: tl.constexpr,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)

    tgt = tl.load(tgt_ptr + offset_nd)
    ignore_mask = tgt != ignore_index
    safe_tgt = tl.where(ignore_mask, tgt, 0)

    if reduction == 0:
        out_grad = tl.load(out_grad_ptr + offset_nd).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if wgt_ptr is None:
        # `ignore_mask.to(tl.float32)` (arith.uitofp on a bool tile) fails
        # TritonXPUUnrollControl verification for BLOCK >= 256 in fp16.
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + safe_tgt).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1.0

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0.0)
    offset_d = offset_nd % D
    offset_n = offset_nd // D
    inp_grad_ptrs = inp_grad_ptr + offset_n * (C * D) + safe_tgt * D + offset_d
    tl.store(inp_grad_ptrs, inp_grad)


# Negative Log Likelihood Loss (NLLLoss)
#
# This loss function is used for training classification problems with C classes.
#
# Parameters:
# - input (Tensor):
#   - Expected to contain log-probabilities for each class.
#   - Shape can be either:
#     - (minibatch, C) for standard classification tasks.
#     - (minibatch, C, d1, d2, ..., dK) for K-dimensional inputs (e.g., per-pixel loss for 2D images).
#
# - target (Tensor):
#   - Should contain class indices in the range [0, C-1].
#   - If ignore_index is specified, this index can be outside the class range
#       and will be ignored in the loss computation.
#
# - weight (1D Tensor, optional):
#   - Assigns weight to each class, useful for unbalanced datasets.
#
# Reduction modes:
# - 'none': returns per-sample loss (shape: (N,)).
# - 'mean' (default): computes the mean of the weighted losses.
# - 'sum': computes the sum of the weighted losses.
#
# Mathematical description:
# - Unreduced loss:
#   l_n = -w_y_n * x_n, where w_c = weight[c] * 1{c != ignore_index}.
# - Reduced loss (depending on the specified reduction mode):
#   - mean: ℓ(x, y) = (1/N) * Σ(w_y_n * l_n)
#   - sum: ℓ(x, y) = Σ(l_n)


# 1d & 2d tensor
def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS_FWD")
    assert self.ndim <= 2, "Invalid input ndim"

    if self.numel() == 0:
        # Empty-input semantics (matches torch): mean->nan, none->empty(target shape), sum->0; total_weight=0
        if reduction == 0:
            loss = torch.empty(target.shape, dtype=self.dtype, device=self.device)
        elif reduction == 1:
            loss = torch.full((), float("nan"), dtype=self.dtype, device=self.device)
        else:
            loss = torch.zeros((), dtype=self.dtype, device=self.device)
        total_weight = torch.zeros((), dtype=self.dtype, device=self.device)
        return loss, total_weight

    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    BLOCK_N = _nll_fwd_block(N)
    # Reduced (mean/sum) paths with a tile-bounded N run the fused
    # single-launch reduction below; that needs both scratch buffers padded to
    # a whole number of unmasked reduction tiles (and to a whole number of
    # elementwise blocks so the pad can be written unmasked).
    fused = False
    if reduction != 0:
        TL = min(_NLL_REDUCE_TILE, triton.next_power_of_2(N))
        ntiles = triton.cdiv(N, TL)
        fused = ntiles <= _NLL_REDUCE_MAX_TILES

    if fused:
        pad_n = triton.cdiv(ntiles * TL, BLOCK_N) * BLOCK_N
        out = torch.empty(pad_n, dtype=self.dtype, device=self.device)
        ignore_weight_tgt = torch.empty(pad_n, dtype=self.dtype, device=self.device)
        n_blocks = pad_n // BLOCK_N
    else:
        out = torch.empty(shape, dtype=self.dtype, device=self.device)
        ignore_weight_tgt = None
        if reduction != 0:
            ignore_weight_tgt = torch.empty(
                target.shape, dtype=self.dtype, device=self.device
            )
        n_blocks = triton.cdiv(N, BLOCK_N)

    with torch_device_fn.device(self.device):
        nll_loss_forward_kernel[(n_blocks, 1, 1)](
            self,  # torch.Size([4096, 256])
            target,  # torch.Size([4096]), tensor([174, 125, 174,  ..., 216, 171, 120])
            weight,  # torch.Size([256])
            out,  # torch.Size([4096])
            ignore_weight_tgt,  # torch.Size([4096])
            ignore_index,  # 1
            N,  # 4096
            C,  # 256
            reduction,  # 0
            BLOCK_N,
            fused,
            is_use_mask_zero=True,
        )

    # redution: 0-None, 1-mean, 2-sum
    if reduction == 0:
        return out, torch.zeros([], dtype=self.dtype, device=self.device)

    if fused:
        output = torch.empty([], dtype=self.dtype, device=self.device)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
        with torch_device_fn.device(self.device):
            nll_loss_reduce_kernel[(1, 1, 1)](
                out,
                ignore_weight_tgt,
                output,
                total_weight,
                reduction == 1,
                ntiles,
                TL,
            )
        return output, total_weight

    if reduction == 1:
        total_out = xpu_sum(out)
        total_weight = xpu_sum(ignore_weight_tgt).to(self.dtype)
        output = (total_out / total_weight).to(self.dtype)
    else:
        total_out = xpu_sum(out)
        output = total_out.to(self.dtype)
        total_weight = xpu_sum(ignore_weight_tgt).to(self.dtype)

    return output, total_weight


def nll_loss_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS_BWD")
    if self.numel() == 0:
        return torch.empty_like(self)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    # The 1d/2d backward is launch-bound: the scatter kernel only touches N of
    # the N*C outputs, so the zero-fill of grad_input dominates the wall clock
    # on every benchmark shape.  `torch.empty_like(...).contiguous()` plus the
    # pointwise_dynamic `zero_` costs ~0.083 ms of fixed dispatch per call on
    # XPU, while the ops/zeros_like.py memset (grid=(12,), buffer_size_limit
    # 2048) does the same work for ~0.045 ms and is never slower on the large
    # shapes either.  Calling it directly (instead of `torch.zeros_like`) keeps
    # the memset on the Kunlunxin gems kernel regardless of the dispatch state
    # and skips one dispatcher round trip.  Measured (fp32, median of 15,
    # XPU 7): 0.147 -> 0.103 ms at (64,64), 0.141 -> 0.099 at (4096,4096),
    # 0.244 -> 0.207 at (1024,65536), 1.63 -> 1.53 at (10000,65536).
    # `zeros_like` preserves `self`'s memory format, so a non-contiguous `self`
    # still takes the old empty_like().contiguous() + zero_ route (the scatter
    # below indexes grad_input row-major as `n * C + tgt`).
    if self.is_contiguous():
        grad_input = xpu_zeros_like(self)
    else:
        grad_input = torch.empty_like(self).contiguous()
        zero_(grad_input)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_backward_kernel[grid](
            grad_output,
            target,
            weight,
            grad_input,
            ignore_index,
            total_weight,
            N,
            C,
            reduction,
        )

    return grad_input


# 3d+ tensor
def nll_loss2d_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS2D_FWD")
    assert self.ndim >= 3, "Invalid input ndim"

    if self.numel() == 0:
        # Empty-input semantics (matches torch): mean->nan, none->empty(target shape), sum->0; total_weight=0
        if reduction == 0:
            loss = torch.empty(target.shape, dtype=self.dtype, device=self.device)
        elif reduction == 1:
            loss = torch.full((), float("nan"), dtype=self.dtype, device=self.device)
        else:
            loss = torch.zeros((), dtype=self.dtype, device=self.device)
        total_weight = torch.zeros((), dtype=self.dtype, device=self.device)
        return loss, total_weight

    N, C = self.shape[0], self.shape[1]
    D = self.numel() // (N * C)
    assert target.numel() == N * D, "Invalid target size"

    target_orig_shape = target.shape
    self_flat = self.reshape(N, C, D).contiguous()
    target_flat = target.reshape(N, D).contiguous()
    weight = None if weight is None else weight.contiguous()

    out = torch.empty((N, D), dtype=self.dtype, device=self.device)
    # `fused` replaces the host-side mean reduction tail (2 x xpu_sum + a 0-d
    # gems div + 2 x .to()) with one extra launch.  Only reduction=mean takes
    # it; see the comment on `_nll2d_fused_reduce`.
    fused = _nll2d_fused_reduce(N * D) if reduction == 1 else None
    ignore_weight_tgt = None
    if reduction == 1:
        ignore_weight_tgt = torch.empty((N, D), dtype=self.dtype, device=self.device)

    block_d = _nll2d_block_d(D)
    flat_block = None if block_d is not None else _nll2d_flat_block(N * D)
    with torch_device_fn.device(self.device):
        if flat_block is not None:
            nll_loss2d_forward_flat_kernel[((N * D) // flat_block, 1, 1)](
                self_flat,
                target_flat,
                weight,
                out,
                ignore_weight_tgt,
                ignore_index,
                C,
                reduction,
                D,
                flat_block,
                is_use_mask_zero=True,
            )
        elif block_d is None:
            grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
            nll_loss2d_forward_kernel[grid](
                self_flat,
                target_flat,
                weight,
                out,
                ignore_weight_tgt,
                ignore_index,
                N,
                C,
                D,
                reduction,
                is_use_mask_zero=True,
            )
        else:
            nll_loss2d_forward_tiled_kernel[(D // block_d, N, 1)](
                self_flat,
                target_flat,
                weight,
                out,
                ignore_weight_tgt,
                ignore_index,
                C,
                reduction,
                D,
                block_d,
                is_use_mask_zero=True,
            )

    # redution: 0-None, 1-mean, 2-sum
    if reduction == 0:
        output = out.reshape(target_orig_shape)
        total_weight = torch.zeros([], dtype=self.dtype, device=self.device)
        return output, total_weight

    if fused is not None:
        ntiles, tl_width = fused
        output = torch.empty([], dtype=self.dtype, device=self.device)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
        with torch_device_fn.device(self.device):
            nll_loss_reduce_kernel[(1, 1, 1)](
                out,
                ignore_weight_tgt,
                output,
                total_weight,
                True,
                ntiles,
                tl_width,
            )
        return output, total_weight

    if reduction == 1:
        total_out = xpu_sum(out)
        total_weight = xpu_sum(ignore_weight_tgt).to(self.dtype)
        output = (total_out / total_weight).to(self.dtype)
    else:
        total_out = xpu_sum(out)
        output = total_out.to(self.dtype)
        total_weight = torch.zeros([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss2d_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS2D_BWD")
    if self.numel() == 0:
        return torch.empty_like(self)
    N, C = self.shape[0], self.shape[1]
    D = self.numel() // (N * C)

    grad_output = grad_output.contiguous()
    target_flat = target.reshape(N, D).contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.empty_like(self).contiguous()
    zero_(grad_input)

    # `flat_block is not None` takes the wide, fully unmasked flat scatter (see
    # the comment on `nll_loss2d_backward_flat_kernel`); everything else keeps
    # HEAD's 128-wide masked kernel byte-for-byte.
    flat_block = _nll2d_bwd_flat_block(N * D)
    with torch_device_fn.device(self.device):
        if flat_block is not None:
            nll_loss2d_backward_flat_kernel[((N * D) // flat_block, 1, 1)](
                grad_output,
                target_flat,
                weight,
                grad_input.reshape(N, C, D),
                ignore_index,
                total_weight,
                C,
                reduction,
                D,
                flat_block,
            )
        else:
            grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
            nll_loss2d_backward_kernel[grid](
                grad_output,
                target_flat,
                weight,
                grad_input.reshape(N, C, D),
                ignore_index,
                total_weight,
                N,
                C,
                D,
                reduction,
            )

    return grad_input


def nll_loss_nd_forward(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
):
    logger.debug("GEMS_KUNLUNXIN NLL LOSS ND FWD")
    if input.numel() == 0:
        # Empty-input semantics (matches torch): mean->nan, none->empty(target shape), sum->0; total_weight=0
        if reduction == 0:
            loss = torch.empty(target.shape, dtype=input.dtype, device=input.device)
        elif reduction == 1:
            loss = torch.full((), float("nan"), dtype=input.dtype, device=input.device)
        else:
            loss = torch.zeros((), dtype=input.dtype, device=input.device)
        total_weight = torch.zeros((), dtype=input.dtype, device=input.device)
        return loss, total_weight
    if input.dim() < 3:
        return nll_loss_forward(
            input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

    return nll_loss2d_forward(
        input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
    )


def nll_loss_nd_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
    total_weight: torch.Tensor = None,
):
    logger.debug("GEMS_KUNLUNXIN NLL LOSS ND BWD")
    if input.numel() == 0:
        return torch.empty_like(input)
    if input.dim() < 3:
        return nll_loss_backward(
            grad_output,
            input,
            target,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            total_weight=total_weight,
        )

    return nll_loss2d_backward(
        grad_output,
        input,
        target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index,
        total_weight=total_weight,
    )
