import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---- tle.raw LM local-histogram O(n) fast path --------------------------------
# Each (cluster,core) GM2LM-strides a chunk of the flattened input into LM and
# does single-core scalar `lh[bin]++` (no vector atomic conflict, no local_ptr
# index drop), then writes its per-core BINS partial once; host reduces partials.
# Verified correct UNDER flag_gems.use_gems() (unlike the scatter per-row long
# write, histc writes each partial once at the end -> no use_gems tail corruption).
# Built lazily to keep backend import side-effect free. Non-adaptive shapes/dtypes
# defer to the codegen kernels below (dispatch, not fallback).
_HISTC_GRID = 12  # this XPU has exactly 12 clusters: grid=16/24 silently drop whole
#                   clusters (partial coverage -> wrong sum), grid=12 covers every
#                   element.  12 clusters x 64 cores = 768 cores vs 512 at grid=8
#                   => 1.32x on the per-core scan chain (device-measured).
_HISTC_CORE_MAX = (
    64  # cores per cluster on this XPU (probed via scatter: GRID*64 covers all)
)
_HISTC_TLE = {"init": False, "ok": False, "k": None}


@triton.jit
def _histc_combine(partial_ptr, out_ptr, bins, TC: tl.constexpr):
    # Cross-core partial reduction: program b sums partial[:, b] over TC rows.
    # Replaces partial.sum(0) (flag_gems reduce ~0.11ms) with one cheap launch.
    b = tl.program_id(0)
    offs = tl.arange(0, TC)
    vals = tl.load(partial_ptr + offs * bins + b)
    tl.store(out_ptr + b, tl.sum(vals, axis=0))


def _histc_tle_init():
    if _HISTC_TLE["init"]:
        return _HISTC_TLE["ok"]
    _HISTC_TLE["init"] = True
    try:
        import os as _os

        import triton as _triton
        import triton.experimental.tle as _tle

        # Precompiled, secrecy-hardened device object (packed from
        # payload/src/histc_lm_raw.xpu via
        # docs/xpu3/how_to_pack_payload/pack_payload.py --obj-dir).  We ship the
        # .o (not the .xpu source): the stub name must equal the entry symbol
        # and the signature must match the C++ ABI.  In object= mode the
        # merge-time signature check is skipped (the payload isn't in IR), so a
        # wrong signature fails at runtime, not compile -- keep the stub in
        # lockstep with the C++.  Re-pack after editing the .xpu: the
        # compile-cache key is the .o content digest, so a stale .o silently
        # keeps the old code.
        _obj = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "payload",
            "obj",
            "histc.o",
        )

        @_tle.raw.dialect("xpu3", object=_obj, arch=3)
        def histc_lm_raw(partial, inp, N, bins, minv, maxv): ...

        @_triton.jit
        def _k(P, INP, N, B, MN, MX):
            _tle.raw.call(histc_lm_raw, (P, INP, N, B, MN, MX))

        _HISTC_TLE["k"] = _k
        _HISTC_TLE["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("GEMS_KUNLUNXIN HISTC tle.raw init failed: %r", _e)
        _HISTC_TLE["ok"] = False
    return _HISTC_TLE["ok"]


def _histc_tle_lm(inp, bins, min_val, max_val):
    """O(n) LM local-histogram via tle.raw; None -> defer to codegen kernels."""
    try:
        if inp.dtype != torch.float32:
            return None
        if not (0 < bins <= 128):
            return None
        if not (max_val > min_val):
            return None
        n = inp.numel()
        if n <= 0 or n >= 2**31:
            return None
        if not _histc_tle_init():
            return None
        flat = inp.contiguous().view(-1)
        tc_alloc = _HISTC_GRID * _HISTC_CORE_MAX
        partial = torch.empty(tc_alloc, bins, dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _HISTC_TLE["k"][(_HISTC_GRID,)](
                partial, flat, n, bins, float(min_val), float(max_val)
            )
        out = torch.empty(bins, dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _histc_combine[(bins,)](partial, out, bins, tc_alloc)
        if out.dtype != inp.dtype:
            out = out.to(inp.dtype)
        return out
    except Exception as _e:  # pragma: no cover - never break correctness
        logger.debug("GEMS_KUNLUNXIN HISTC tle path deferred: %r", _e)
        return None


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
    b = ext.program_id(0)
    inv_scale = bins / (max_val - min_val)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for start in range(0, n_main, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        v = tl.load(inp_ptr + offs).to(tl.float32)
        idx = tl.floor((v - min_val) * inv_scale).to(tl.int32)
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
            inp,
            mn_part,
            mx_part,
            n,
            TILE=8192,
            GRID=grid,
        )
        g_pow = 1
        while g_pow < grid:
            g_pow *= 2
        kern_range_combine[(1,)](
            mn_part,
            mx_part,
            out_mn,
            out_mx,
            G=g_pow,
            NG=grid,
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
        ones = torch.full((1,), count, dtype=inp.dtype, device=inp.device)
        torch.ops.aten._copy_from(ones, out[bins // 2 : bins // 2 + 1], False)
        return out

    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    for _bs in (8192, 4096, 2048, 1024):
        if n_elements % _bs == 0:
            BLOCK_SIZE = _bs
            break
    grid = (bins,)

    n_main = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    n_tail = n_elements - n_main

    _tle_out = _histc_tle_lm(inp, bins, min_val, max_val)
    if _tle_out is not None:
        return _tle_out

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
