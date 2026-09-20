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
