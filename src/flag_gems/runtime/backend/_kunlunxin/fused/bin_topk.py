import sys

import torch
import triton
import triton.language as tl

BS = 8192
RANK_SUB = 512
RANK_NSUB = BS // RANK_SUB


@triton.jit
def _ord_i32(x):
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(x >= 0, bits | (-0x80000000), ~bits)


@triton.jit
def _uge(a, b):
    return (a ^ (-0x80000000)) >= (b ^ (-0x80000000))


@triton.jit
def _ugt(a, b):
    return (a ^ (-0x80000000)) > (b ^ (-0x80000000))


@triton.jit
def _bst_count_kernel(
    inputs, starts, ends, thrs, cnts, k_eff, bit, S: tl.constexpr, BS: tl.constexpr
):
    """Count elements >= (thr | (1 << bit)) per row, then update thrs in
    place: thrs[b] = (cnt >= k_eff[b]) ? cand : thrs[b].  Fuses the host-side
    ``cands = thrs | (one << bit)`` and ``torch.where`` of the bitwise
    radix-select loop into one kernel."""
    b = tl.program_id(0)
    s_base = inputs + b * S
    start = tl.load(starts + b).to(tl.int32)
    end = tl.load(ends + b).to(tl.int32)
    n = end - start
    TS = tl.cdiv(n, BS)
    thr = tl.load(thrs + b)
    cand = thr | (1 << bit)
    cnt = 0
    for t in range(TS):
        offs = t * BS + tl.arange(0, BS)
        m = offs < n
        x = tl.load(s_base + start + offs, mask=m, other=0.0).to(tl.float32)
        u = _ord_i32(x)
        cnt += tl.sum((m & _uge(u, cand)).to(tl.int32), axis=0)
    tl.store(cnts + b, cnt)
    tl.store(thrs + b, tl.where(cnt >= tl.load(k_eff + b), cand, thr))


@triton.jit
def _bst_rank_kernel(
    inputs,
    ranks,
    starts,
    ends,
    thrs,
    S: tl.constexpr,
    RB: tl.constexpr,
    SUB: tl.constexpr,
    NSUB: tl.constexpr,
):
    b = tl.program_id(0)
    s_base = inputs + b * S
    rank_base = ranks + b * RB
    start = tl.load(starts + b).to(tl.int32)
    end = tl.load(ends + b).to(tl.int32)
    n = end - start
    TS = tl.cdiv(n, SUB)
    thr = tl.load(thrs + b)
    prev_gt = 0
    prev_eq = 0
    for t in range(TS):
        offs = t * SUB + tl.arange(0, SUB)
        m = offs < n
        x = tl.load(s_base + start + offs, mask=m, other=0.0).to(tl.float32)
        u = _ord_i32(x)
        gt = _ugt(u, thr) & m
        eq = (u == thr) & m
        cums_gt = tl.cumsum(gt.to(tl.int32), axis=0)
        cums_eq = tl.cumsum(eq.to(tl.int32), axis=0)
        val = (cums_gt + prev_gt) * gt.to(tl.int32) - (cums_eq + prev_eq) * eq.to(
            tl.int32
        )
        tl.store(rank_base + offs, val, mask=m)
        prev_gt += tl.sum(gt.to(tl.int32), axis=0)
        prev_eq += tl.sum(eq.to(tl.int32), axis=0)


@triton.jit
def _bst_fill_kernel(
    ranks,
    out,
    scratch,
    n_arr,
    starts,
    gmap,
    K: tl.constexpr,
    BSF: tl.constexpr,
    RB: tl.constexpr,
    HAS_GMAP: tl.constexpr,
):
    b = tl.program_id(0)
    rank_base = ranks + b * RB
    out_base = out + b * K
    scr = scratch + b * (K + 4)
    n = tl.load(n_arr + b)
    st = tl.load(starts + b).to(tl.int32)
    offs = tl.arange(0, BSF)
    m = offs < n
    rk = tl.load(rank_base + offs, mask=m, other=0)
    if HAS_GMAP:
        lane = tl.load(gmap + st + offs, mask=m, other=-1)
    else:
        lane = offs + st
    c1 = tl.sum((rk > 0).to(tl.int32), axis=0)
    c2 = tl.sum((rk < 0).to(tl.int32), axis=0)
    for p in range(0, c1):
        v = tl.sum((rk == p + 1).to(tl.int32) * lane, axis=0)
        tl.store(scr + p, v)
        tl.store(out_base + p, v)
    for q in range(0, tl.minimum(c2, K - c1)):
        v = tl.sum((rk == -(q + 1)).to(tl.int32) * lane, axis=0)
        tl.store(scr + K + q, v)
        tl.store(out_base + c1 + q, v)


@triton.jit
def _bst_gather_vals_kernel(x, cidx, cval, K: tl.constexpr):
    j = tl.program_id(0)
    off = j * K + tl.arange(0, K)
    idx = tl.load(cidx + off)
    m = idx >= 0
    idx_c = tl.where(m, idx, 0)
    v = tl.load(x + idx_c).to(tl.float32)
    v = tl.where(m, v, float("-inf"))
    tl.store(cval + off, v)


@triton.jit
def _bst_map_idx_kernel(gmap, idx, out, N: tl.constexpr):
    j = tl.program_id(0)
    off = j * 1024 + tl.arange(0, 1024)
    m = off < N
    i = tl.load(idx + off, mask=m, other=0)
    im = m & (i >= 0)
    ic = tl.where(im, i, 0)
    v = tl.load(gmap + ic)
    tl.store(out + off, tl.where(im, v, -1), mask=m)


def _bst_select(x, starts, ends, n, k_eff, K, out, gmap=None):
    """3-kernel pipeline. x: (B, S). starts/ends/n/k_eff: (B,). out: (B, K)."""
    Bb = x.shape[0]
    thrs = torch.zeros(Bb, dtype=torch.int32, device=x.device)
    cnts = torch.empty(Bb, dtype=torch.int32, device=x.device)
    for bit in range(31, -1, -1):
        _bst_count_kernel[(Bb,)](
            x,
            starts,
            ends,
            thrs,
            cnts,
            k_eff,
            bit,
            x.shape[1],
            BS,
            num_warps=4,
            num_stages=1,
        )
    ranks = torch.zeros(Bb, BS, dtype=torch.int32, device=x.device)
    _bst_rank_kernel[(Bb,)](
        x,
        ranks,
        starts,
        ends,
        thrs,
        x.shape[1],
        BS,
        RANK_SUB,
        RANK_NSUB,
        num_warps=4,
        num_stages=1,
    )
    scratch = torch.zeros(Bb, 2 * K + 8, dtype=torch.int32, device=x.device)
    if gmap is None:
        gm = torch.zeros(1, dtype=torch.int32, device=x.device)
        _bst_fill_kernel[(Bb,)](
            ranks,
            out,
            scratch,
            n,
            starts,
            gm,
            K,
            BS,
            BS,
            False,
            num_warps=4,
            num_stages=1,
        )
    else:
        _bst_fill_kernel[(Bb,)](
            ranks,
            out,
            scratch,
            n,
            starts,
            gmap,
            K,
            BS,
            BS,
            True,
            num_warps=4,
            num_stages=1,
        )


def _bst_rows(xv, st_val, en_val, n_val, K, out, gmap=None):
    """Recursive chunked select on a single row. xv: (1, S) values;
    st_val/en_val: (1,) int32 row [start, end); gmap: (S,) global index map
    or None; out: (1, K) global indices."""
    nn = int(n_val[0].item())
    if nn <= 0:
        return
    st0 = int(st_val[0].item())
    starts = torch.tensor([st0], dtype=torch.int32, device=xv.device)
    if nn <= BS:
        n1 = torch.tensor([nn], dtype=torch.int32, device=xv.device)
        k1 = torch.tensor([min(K, nn)], dtype=torch.int32, device=xv.device)
        _bst_select(xv, starts, en_val, n1, k1, K, out, gmap)
        return
    nch = (nn + BS - 1) // BS
    cidx = torch.full((nch, K), -1, dtype=torch.int32, device=xv.device)
    for j in range(nch):
        cst = st0 + j * BS
        cn = min(BS, nn - j * BS)
        if cn <= 0:
            continue
        st_c = torch.tensor([cst], dtype=torch.int32, device=xv.device)
        en_c = torch.tensor([cst + cn], dtype=torch.int32, device=xv.device)
        n_c = torch.tensor([cn], dtype=torch.int32, device=xv.device)
        k_c = torch.tensor([min(K, cn)], dtype=torch.int32, device=xv.device)
        _bst_select(xv, st_c, en_c, n_c, k_c, K, cidx[j : j + 1])
    cflat = cidx.reshape(-1)
    M = nch * K
    cvals = torch.full((M,), float("-inf"), device=xv.device)
    _bst_gather_vals_kernel[(nch,)](xv[0], cflat, cvals, K, num_warps=4, num_stages=1)
    if gmap is None:
        gm = cflat
    else:
        gm = torch.full((M,), -1, dtype=torch.int32, device=xv.device)
        _bst_map_idx_kernel[((M + 1023) // 1024,)](
            gmap, cflat, gm, M, num_warps=4, num_stages=1
        )
    z1 = torch.tensor([0], dtype=torch.int32, device=xv.device)
    n1 = torch.tensor([M], dtype=torch.int32, device=xv.device)
    _bst_rows(cvals.view(1, M), z1, n1, n1, K, out, gm)


def bucket_sort_topk_xpu(inputs, starts, ends, topk):
    x = inputs.float() if inputs.dtype != torch.float32 else inputs
    B, S = x.shape
    K = topk
    out = torch.full((B, K), -1, dtype=torch.int32, device=x.device)
    if B == 0 or S == 0:
        return out
    n = (ends - starts).to(torch.int32)
    with torch.no_grad():
        for b in range(B):
            _bst_rows(
                x[b : b + 1],
                starts[b : b + 1],
                ends[b : b + 1],
                n[b : b + 1],
                K,
                out[b : b + 1],
            )
    return out


def _install():
    """Replace ``flag_gems.fused.DSA.bin_topk.bucket_sort_topk`` with the XPU
    implementation. The DSA bin_topk family is called via direct module
    import (``from flag_gems.fused.DSA.bin_topk import bucket_sort_topk``) in
    tests/test_DSA/test_bin_topk.py, so the SpecOpRegistrar namespace swap
    cannot reach it; the attribute of the already-imported module (loaded
    during ``import flag_gems``) is patched here instead.

    ``gmod.HAS_TLE`` is also set to True: the kunlunxin VendorDescriptor keeps
    ``tle_enabled=False`` (the generic TLE kernel does not lower on the
    Triton-XPU backend), so the module-level ``HAS_TLE`` guard in
    tests/test_DSA/test_bin_topk.py would otherwise skip the whole
    ``bucket_sort_topk`` matrix even though the vendor replacement above is a
    complete, functional implementation of the operator. The only other
    readers of ``flag_gems.fused.DSA.bin_topk.HAS_TLE`` are
    ``tle_bucket_sort_topk`` / ``_should_use_tle_bucket_sort_topk``, which are
    only reachable through the (now replaced) generic ``bucket_sort_topk``
    entrypoint, so no TLE code path is activated on XPU."""
    gmod = sys.modules.get("flag_gems.fused.DSA.bin_topk")
    if gmod is not None:
        gmod.bucket_sort_topk = bucket_sort_topk_xpu
        gmod.HAS_TLE = True


_install()
