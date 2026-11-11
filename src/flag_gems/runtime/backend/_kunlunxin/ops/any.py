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
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value

# ---- perf design (2026-09-05, measured on XPU dev6) ----
# The per-row reduction kernel uses a persisted [BLOCK_M, BLOCK_N] accumulator with a
# SINGLE axis=1 reduce after the loop (same skeleton as mean_dim). What changed vs the
# old int1 OR-tree / int32-word / fp32-max paths:
#   * reduce-op cost dominates on XPU, NOT bandwidth:
#       fp32 max/min  ~535GB/s   (fast)
#       fp32 add      ~547GB/s   (fast)
#       int1 OR/AND   ~97GB/s    (slow)
#       int32 max     ~90-183GB/s(slow)
#       int bitcast / mask+bitcast ~30GB/s (very slow)
#   => drop the int-word bitmap for ANY entirely; use native fp32 max(|x|).
#   * native-dtype accumulate: fp16 native max ~253GB/s vs fp16->fp32 ~204GB/s.
#     bf16 tl.maximum promotes to fp32 on XPU, so bf16 always accumulates in fp32.
#   * tile [64, 512] = 32768 lanes is the sweet spot (no spill, ~535GB/s fp32).
# Semantics: any(row) = (max(|row|) != 0). This is exactly `any(x != 0)` for real
# values (NaN -> |NaN|=NaN -> !=0 -> True; +/-0 -> 0 -> False). Same result as the
# old max path for the {0,1}/zeros/ones test inputs.

BLOCK_M_DEFAULT = 64
BLOCK_N_DEFAULT = 512


def _acc_dtype(dt):
    # fp16 stays native (faster); bf16 (and everything else) accumulates in fp32
    return tl.float16 if dt == torch.float16 else tl.float32


def _row_tile_rows(args):
    """BLOCK_M for the row reduction: 128 for fp16, 64 for everything else.

    Evidence (node97, do_bench metric, 3 interleaved runs, ±1% spread; per-tile check against
    native torch.any correct): on the [4096,4096] row reduction, raising fp16 BLOCK_M from
    64 to 128 takes the kernel from 126.9µs to 94.5µs (−26%, 127.2 → 99.7µs for that cell
    under the official metric); f32/bf16 stay at 64 (128 rows push the fp32 accumulator to
    256KB and are actually slower at 121/205µs vs 114/202µs). In physical terms: keep the
    accumulator tile at ~128KB -- fp16 at 2B/lane allows 128 rows, fp32 at 4B/lane only 64.
    """
    t = args.get("inp")
    if getattr(t, "dtype", None) == torch.float16:
        return BLOCK_M_DEFAULT * 2
    return BLOCK_M_DEFAULT


def heur_m_block_size(args):
    return min(triton.next_power_of_2(args["M"]), _row_tile_rows(args))


def heur_n_block_size(args):
    return min(triton.next_power_of_2(args["N"]), BLOCK_N_DEFAULT)


def heur_m_block_size_p(args):
    return min(triton.next_power_of_2(args["P"]), BLOCK_M_DEFAULT)


def heur_n_block_size_p(args):
    return min(triton.next_power_of_2(args["C"]), BLOCK_N_DEFAULT)


def heur_n_block_size_nw(args):
    return min(triton.next_power_of_2(args["NW"]), BLOCK_N_DEFAULT)


@triton.jit
def _max2(a, b):
    return tl.maximum(a, b)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def any_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
    EVEN: tl.constexpr = False,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * N
    outb = out + rows
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC)
    if EVEN:
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            a = tl.load(inb + cols).to(ACC)
            acc = tl.maximum(acc, tl.abs(a))
        r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
        tl.store(outb, r != 0)
    else:
        row_mask = rows < M
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = row_mask and (cols < N)
            a = tl.load(inb + cols, mask, other=0.0).to(ACC)
            acc = tl.maximum(acc, tl.abs(a))
        r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
        tl.store(outb, r != 0, row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size_nw,
    },
)
@triton.jit
def any_bool_dim_kernel(
    inw,
    out,
    M,
    NW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-row ANY on a bool tensor viewed as int32 words (NW = N//4 words per row).

    word != 0 <=> at least one of its 4 bools is 1 (bool bytes are 0x00/0x01, so a
    zero word is exactly four zero bools). int32 max -> (max != 0)."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * NW
    outb = out + rows
    row_mask = rows < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < NW)
        w = tl.load(inb + cols, mask, other=0)
        acc = tl.maximum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
    tl.store(outb, r != 0, row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def any_dim_kernel_f(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
    EVEN: tl.constexpr = False,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * N
    outb = out + rows
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC)
    if EVEN:
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            a = tl.load(inb + cols).to(ACC)
            acc = tl.maximum(acc, tl.abs(a))
        r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
        tl.store(outb, r)
    else:
        row_mask = rows < M
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = row_mask and (cols < N)
            a = tl.load(inb + cols, mask, other=0.0).to(ACC)
            acc = tl.maximum(acc, tl.abs(a))
        r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
        tl.store(outb, r, row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size_nw,
    },
)
@triton.jit
def any_bool_dim_kernel_f(
    inw,
    out,
    M,
    NW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """any_bool_dim_kernel variant storing the raw reduced value (see -f note).

    Note: accumulate in **fp32**, not int32: int32 max is markedly slower on this backend --
    [4096,4096] measured 120.1µs → 63.3µs (−47%, each checked against native
    torch.any, all correct). word ≥ 0 ⇒ the fp32 value stays
    non-zero (< 2^24 it is exact, beyond that round-to-nearest is still non-zero), so the
    semantics are unchanged."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * NW
    outb = out + rows
    row_mask = rows < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < NW)
        w = tl.load(inb + cols, mask, other=0)
        acc = tl.maximum(acc, w.to(tl.float32))
    r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
    tl.store(outb, r, row_mask)


# ---- global (all elements reduced to a single bool): two-stage ----
# Stage 1 views the flat buffer as [P, C] and reduces each of the P chunk-rows
# (grid = cdiv(P, BLOCK_M), the same fast per-row tile). Stage 2 reduces the P
# partials in one program. P is chosen as a divisor of n (falls back to P=1).
_GLOBAL_CHUNKS = (256, 128, 64, 32, 16, 8, 4, 2, 1)


def _pick_chunks(n):
    for p in _GLOBAL_CHUNKS:
        if n % p == 0:
            return p
    return 1


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size_p,
        "BLOCK_N": heur_n_block_size_p,
    },
)
@triton.jit
def any_global_s1(
    inp,
    mid,
    P,
    C,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * C
    midb = mid + rows
    row_mask = rows < P
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < C)
        a = tl.load(inb + cols, mask, other=0.0).to(ACC)
        acc = tl.maximum(acc, tl.abs(a))
    r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
    tl.store(midb, r, row_mask)


@libentry()
@triton.jit
def any_global_s2(mid, out, P, BLOCK: tl.constexpr, ACC: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < P
    a = tl.load(mid + offs, mask=mask, other=0.0).to(ACC)
    r = tl.reduce(a, axis=0, combine_fn=_max2)
    tl.store(out, r != 0)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size_p,
        "BLOCK_N": heur_n_block_size_p,
    },
)
@triton.jit
def any_global_bool_s1(inw, mid, P, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * C
    midb = mid + rows
    row_mask = rows < P
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < C)
        w = tl.load(inb + cols, mask, other=0)
        acc = tl.maximum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_max2)[:, None]
    tl.store(midb, r, row_mask)


@libentry()
@triton.jit
def any_global_bool_s2(mid, out, P, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < P
    a = tl.load(mid + offs, mask=mask, other=0)
    r = tl.reduce(a, axis=0, combine_fn=_max2)
    tl.store(out, r != 0)


def _global_any(inp):
    """Reduce a flat contiguous input to a single bool. `inp` must be contiguous."""
    n = inp.numel()
    out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)

    if inp.dtype == torch.bool and n % 4 == 0:
        view = inp.reshape(-1).view(torch.int32)
        nw = view.numel()
        p = _pick_chunks(nw)
        c = nw // p
        mid = torch.empty((p,), dtype=torch.int32, device=inp.device)
        with torch_device_fn.device(inp.device):
            any_global_bool_s1[(triton.cdiv(p, BLOCK_M_DEFAULT), 1)](
                view.reshape(p, c), mid, p, c, buffer_size_limit=2048
            )
            if p == 1:
                return (mid[0] != 0).reshape([])
            any_global_bool_s2[(1, 1)](
                mid, out, p, triton.next_power_of_2(p), buffer_size_limit=2048
            )
        return out

    # Non-bool: reduce the native elements directly (native-dtype accumulate).
    p = _pick_chunks(n)
    c = n // p
    acc = _acc_dtype(inp.dtype)
    mid = torch.empty((p,), dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        any_global_s1[(triton.cdiv(p, BLOCK_M_DEFAULT), 1)](
            inp.reshape(p, c), mid, p, c, ACC=acc, buffer_size_limit=2048
        )
        if p == 1:
            return (mid[0] != 0).reshape([])
        any_global_s2[(1, 1)](
            mid, out, p, triton.next_power_of_2(p), ACC=acc, buffer_size_limit=2048
        )
    return out


@triton.jit
def reduce_any(a, b):
    return a or b


@libentry()
@triton.jit
def any_elem_s1(inp, mid, n, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(inp + offs, mask=mask, other=0)
    nz = tl.where(mask, v != 0, False)
    r = tl.reduce(nz, axis=0, combine_fn=reduce_any)
    tl.store(mid + pid, r)


@libentry()
@triton.jit
def any_elem_s2(mid, out, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(mid + offs, mask=mask, other=0)
    r = tl.reduce(v, axis=0, combine_fn=reduce_any)
    tl.store(out, r)


def any(inp):
    logger.debug("GEMS_KUNLUNXIN ANY")
    if inp.is_contiguous():
        return _global_any(inp)
    n = inp.numel()
    out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
    block_size = 2048
    mid_size = triton.cdiv(n, block_size)
    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        any_elem_s1[(mid_size, 1)](inp, mid, n, block_size, buffer_size_limit=2048)
        if mid_size == 1:
            return mid.reshape([])
        any_elem_s2[(1, 1)](
            mid, out, mid_size, triton.next_power_of_2(mid_size), buffer_size_limit=2048
        )
    return out


@libentry()
@triton.jit
def any_mid_neq0_kernel(mid, out, M, BLOCK: tl.constexpr):
    """`mid[M] -> out[M]` (bool, `v != 0`) -- the second stage of the two-step path.

    Why not `(mid != 0)`: inside the `use_gems()` context that is a **nested gem dispatch**
    (the op dispatches to a gem again internally); under the official metric the A/B on
    bool [4096,4096] measured 64.9µs (this kernel) vs 76.7µs (`(mid != 0)`) -- ~12µs apart.
    This kernel performs the same elementwise test in a single launch of its own.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    v = tl.load(mid + offs, mask=mask, other=0)
    tl.store(out + offs, v != 0, mask=mask)


# ---- N-split (row-chunk) parallelism ----
# The per-row kernel above maps ONE program tile onto consecutive rows, so its
# grid is cdiv(M, BLOCK_M): when M is small (e.g. any_dims over several leading
# dims leaves a tiny output) the device is starved (measured: 2 CTAs, 20 GB/s).
# Splitting every reduced row into `S` equal chunks restores parallelism without
# a new kernel: chunk `j` of row `m` is exactly the row `m*S+j` of the
# `[M*S, N/S]` reshape of the same contiguous buffer, and the S partials are then
# folded by the same row kernel with `S` columns.
_SPLIT_FACTORS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
_SPLIT_MIN_CHUNK = 128  # keep >= 128 elements per partial row
_SPLIT_MIN_UNITS = 4096  # below this the extra launches never pay off
_SPLIT_MIN_TOTAL = 1 << 20  # per-row path: minimum elements to split
_SPLIT_MIN_TOTAL_GLOBAL = 4 << 20  # global path: `_global_any` is already wide
_SPLIT_MAX_PROGRAMS = 4  # only split when the single-pass grid is this starved


def _split_target_rows(M, N):
    """Partial-stage row count (~CTA count x BLOCK_M) that measured fastest.

    Bulk tensors amortize the fold and profit from many more CTAs (XPU1,
    2026-10-21: (100,65536,100) fp16 6.53 ms at 16384 rows vs 8.27 ms at 2048;
    (100,256,100) fp16 0.10 ms at 2048 vs 0.13 ms at 16384).
    """
    total = M * N
    if M == 1 or total >= (32 << 20):
        return 16384
    if total >= (8 << 20):
        return 4096
    return 2048


def _pick_split(M, N, block_m):
    """Chunks to split each reduced row into; 1 means no split."""
    if N < _SPLIT_MIN_UNITS:
        return 1
    target = _split_target_rows(M, N)
    if M >= target:
        return 1
    # A split costs two extra launches plus a fold; it only pays for bulk data.
    # The global case (M == 1) keeps the chunked `_global_any` staging, which is
    # already several CTAs wide, so it needs more work to justify the trips.
    total = M * N
    if M == 1:
        if total < _SPLIT_MIN_TOTAL_GLOBAL:
            return 1
    else:
        if total < _SPLIT_MIN_TOTAL:
            return 1
        # Measured (XPU1, 2026-10-21): with >= 8 row programs the single pass is
        # already bandwidth-bound (fp32 (64,512,512): 8 CTA reach 350 GB/s and the
        # split is a 7-13% regression); 1-4 CTA is where the device is starved
        # (fp16 (100,256,100): 1 CTA, 18 GB/s).
        if (M + block_m - 1) // block_m > _SPLIT_MAX_PROGRAMS:
            return 1
    best = 1
    for s in _SPLIT_FACTORS:
        if N // s < _SPLIT_MIN_CHUNK:
            break
        if N % s:
            continue
        best = s
        if M * s >= target:
            return s
    return best if M * best >= 4 * BLOCK_M_DEFAULT else 1


def _row_block_m(rows, dtype):
    """Host-side mirror of `heur_m_block_size` (keeps the launch grid a constant
    tuple; a `grid = lambda meta: ...` built inside a function body re-enters the
    XPU compilation cache key on every call)."""
    per_row = BLOCK_M_DEFAULT * 2 if dtype == torch.float16 else BLOCK_M_DEFAULT
    return min(triton.next_power_of_2(rows), per_row)


def _even_flag(rows, cols, dtype):
    """Host mirror of the kernels' `EVEN` constexpr: True only when every access
    mask is statically all-true (host-side twins of `heur_m_block_size` /
    `heur_n_block_size`), which lets the kernel drop the masked-memory path.

    Only pays off on wide tiles: dropping the mask on a narrow one (BLOCK_N below
    128, e.g. the N=1 / N=16 cases) measures *slower* (XPU1 2026-10-21: fp16
    (1024,1) 19.0 us masked vs 31.3 us mask-free), so those keep the masked path.
    """
    block_n = min(triton.next_power_of_2(cols), BLOCK_N_DEFAULT)
    return (
        block_n >= 128 and rows % _row_block_m(rows, dtype) == 0 and cols % block_n == 0
    )


def _per_row_any(inp, M, N, out_shape, allow_split=False):
    """Reduce a contiguous [M, N] view over its N axis (per row) -> bool tensor.

    For large M the bool output is produced in two steps (reduce -> float/int
    scratch, then `scratch != 0`): a 1-byte output store scalarizes the whole
    kernel on XPU (~2.3x at [4096,4096]), so writing a wider result and doing a
    tiny follow-up compare is faster. Small M stays single-kernel (launch-bound).

    `allow_split=True` enables the N-split path (see `_pick_split`): a partial
    row reduction followed by a fold of the S partials. It is only used by
    `any_dims`; `any_dim` keeps the original single-pass behaviour."""
    is_bool_words = inp.dtype == torch.bool and N % 4 == 0
    units = N // 4 if is_bool_words else N
    block_m = _row_block_m(M, inp.dtype)
    split = _pick_split(M, units, block_m) if allow_split else 1
    if M == 1 and split == 1 and allow_split:
        # Single-row reduce with no profitable split: the two-stage global path
        # (`any`) is several CTAs wide and avoids one long single-CTA loop.
        return any(inp).reshape(out_shape)
    if split > 1:
        rows = M * split
        chunk = units // split
        if is_bool_words:
            src = inp.reshape(-1).view(torch.int32).reshape(rows, chunk)
            part = torch.empty(rows, dtype=torch.float32, device=inp.device)
            grid1 = (triton.cdiv(rows, _row_block_m(rows, src.dtype)),)
            with torch_device_fn.device(inp.device):
                any_bool_dim_kernel_f[grid1](
                    src, part, rows, chunk, buffer_size_limit=2048
                )
        else:
            src = inp.reshape(rows, chunk)
            acc = _acc_dtype(inp.dtype)
            mid_dt = torch.float16 if acc == tl.float16 else torch.float32
            part = torch.empty(rows, dtype=mid_dt, device=inp.device)
            grid1 = (triton.cdiv(rows, _row_block_m(rows, src.dtype)),)
            with torch_device_fn.device(inp.device):
                any_dim_kernel_f[grid1](
                    src,
                    part,
                    rows,
                    chunk,
                    ACC=acc,
                    EVEN=_even_flag(rows, chunk, src.dtype),
                    buffer_size_limit=2048,
                )
        out = torch.empty(M, dtype=torch.bool, device=inp.device)
        grid2 = (triton.cdiv(M, _row_block_m(M, part.dtype)),)
        with torch_device_fn.device(inp.device):
            # The fold sees only S columns per row (tiny), so the cheap bool-store
            # variant is used directly instead of the two-step `-f` + neq0 pair.
            any_dim_kernel[grid2](
                part.view(M, split),
                out,
                M,
                split,
                ACC=tl.float32,
                EVEN=_even_flag(M, split, part.dtype),
                buffer_size_limit=2048,
            )
        return out.reshape(out_shape)

    grid = (triton.cdiv(M, block_m),)
    mid_grid = (triton.cdiv(M, 2048),)
    # The two-step form exists because a 1-byte result store scalarizes a wide
    # kernel; for a handful of rows the extra launch costs more than it saves.
    two_step = M > BLOCK_M_DEFAULT and (not allow_split or M * N >= 65536)
    if is_bool_words:
        inw = inp.reshape(-1).view(torch.int32).reshape(M, N // 4)
        if two_step:
            # mid follows the fp32 accumulation of `any_bool_dim_kernel_f` (see that kernel's comment)
            mid = torch.empty(M, dtype=torch.float32, device=inp.device)
            out = torch.empty(M, dtype=torch.bool, device=inp.device)
            with torch_device_fn.device(inp.device):
                any_bool_dim_kernel_f[grid](inw, mid, M, N // 4, buffer_size_limit=2048)
                any_mid_neq0_kernel[mid_grid](
                    mid, out, M, BLOCK=2048, buffer_size_limit=2048
                )
            return out.reshape(out_shape)
        out = torch.empty(M, dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            any_bool_dim_kernel[grid](inw, out, M, N // 4, buffer_size_limit=2048)
    else:
        acc = _acc_dtype(inp.dtype)
        if two_step:
            mid_dt = torch.float16 if acc == tl.float16 else torch.float32
            mid = torch.empty(M, dtype=mid_dt, device=inp.device)
            out = torch.empty(M, dtype=torch.bool, device=inp.device)
            with torch_device_fn.device(inp.device):
                any_dim_kernel_f[grid](
                    inp,
                    mid,
                    M,
                    N,
                    ACC=acc,
                    EVEN=_even_flag(M, N, inp.dtype),
                    buffer_size_limit=2048,
                )
                any_mid_neq0_kernel[mid_grid](
                    mid, out, M, BLOCK=2048, buffer_size_limit=2048
                )
            return out.reshape(out_shape)
        out = torch.empty(M, dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            any_dim_kernel[grid](
                inp,
                out,
                M,
                N,
                ACC=acc,
                EVEN=_even_flag(M, N, inp.dtype),
                buffer_size_limit=2048,
            )
    return out.reshape(out_shape)


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ANY_DIM")
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        if dim != inp.ndim - 1 or not inp.is_contiguous():
            inp = dim_compress(inp, dim)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        if M == 1:
            # All elements are on the single reduced row -> a global any().
            out = any(inp).reshape(shape)
        else:
            out = _per_row_any(inp, M, N, shape)

        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def _compress_dims(inp, dim):
    if (
        inp.dtype == torch.bool
        and inp.is_contiguous()
        and inp.ndim > 0
        and inp.shape[-1] > 0
        and inp.shape[-1] % 4 == 0
    ):
        # Skip the packing when the compression is a no-op anyway (2-D `[0,1]`):
        # the extra view/reshape dispatches are real cost at 10-20 us scale.
        stride = inp.stride()
        batch_dim = [i for i in range(inp.ndim) if i not in dim]
        order = batch_dim + sorted(dim, key=lambda x: stride[x], reverse=True)
        if order != list(range(inp.ndim)):
            packed = dim_compress(inp.view(torch.int32), dim)
            return packed.view(torch.bool).reshape(
                packed.shape[:-1] + (packed.shape[-1] * 4,)
            )
    return dim_compress(inp, dim)


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ANY_DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    # bool goes through the packed staging helper; everything else is a plain
    # `dim_compress` (kept call-compatible with the pre-change path).
    inp = (
        _compress_dims(inp, dim) if inp.dtype == torch.bool else dim_compress(inp, dim)
    )
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if M == 1:
        # Cheap mirror of `_pick_split(1, units, ...) == 1` so the common global
        # case keeps the exact pre-change call chain (`any`), no helper hops.
        units = N // 4 if (inp.dtype == torch.bool and N % 4 == 0) else N
        if units >= _SPLIT_MIN_UNITS and units >= _SPLIT_MIN_TOTAL_GLOBAL:
            out = _per_row_any(inp, M, N, shape, allow_split=True)
        else:
            out = any(inp).reshape(shape)
    else:
        out = _per_row_any(inp, M, N, shape, allow_split=True)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
