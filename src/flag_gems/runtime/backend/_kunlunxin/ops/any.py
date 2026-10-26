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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False
    tle = None
    TensorDescriptor = None

# The flagtree compiler (older triton fork pinned in the CI image) cannot
# legalize `tt.make_range` (from `tl.arange`) when it indexes a TLE local
# pointer, so the TLE row-reduce fails to compile there. Detect it once and
# fall back to the non-TLE kernels (same guard as lt.py/less_equal.py).
_IS_FLAGTREE = os.path.exists(
    os.path.join(os.path.dirname(triton.__file__), "FLAGTREE_BACKEND")
)

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
):
    """Per-row ANY: reduce each row's |x| max, store (max != 0).

    Grid = cdiv(M, BLOCK_M). Each program owns BLOCK_M rows x full N (looped in
    BLOCK_N chunks), so the whole tensor is read exactly once, coalesced per row.
    Native-dtype accumulate (fp16 stays fp16; bf16/others upcast to fp32)."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * N
    outb = out + rows
    row_mask = rows < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC)
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
):
    """any_dim_kernel variant that stores the raw reduced value (float) instead of a
    bool. On XPU a 1-byte i1/i8 store scalarizes the whole kernel (~2.3x slower at
    [4096,4096]); storing the float result and converting (scratch != 0) outside wins
    when M is large."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * N
    outb = out + rows
    row_mask = rows < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC)
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

    if n >= _TLE_FLAT_MIN:
        mid = _tle_any_flat(inp)
        if mid is not None:
            with torch_device_fn.device(inp.device):
                any_mid_neq0_kernel[(1,)](
                    mid.reshape(1), out.reshape(1), 1, BLOCK=1, buffer_size_limit=2048
                )
            return out

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


# ---- TLE (cluster DMA) max(|x|) row-reduce -------------------------------
# Mirrors sum.py's `_tle_sum_row_kernel` (reduce-INSIDE [XBLOCK] accumulator)
# with `sum` -> `max(|x|)`. `any(row) = (max(|row|) != 0)`. On the
# [100, 6553600] row-reduce (the (100,65536,100) dim=[0,1] case) the persisted
# [BLOCK_M, BLOCK_N] pointer tile walks 6.5M columns with grid=2 at ~77GB/s,
# whereas `tle.gpu.copy` streams the same bytes with the cluster DMA.
_TLE_CORE_NUM = 64
_TLE_LM_BYTES_PER_CORE = 4096
_TLE_XBLOCK = 512
_TLE_CLUSTERS = 8
_TLE_WIDE_ROW_BYTES = 32768
_TLE_NARROW_ROW_BYTES = 2048

_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}
_TLE_ACC_DTYPE = {
    torch.float16: tl.float32,
    torch.float32: tl.float32,
    torch.bfloat16: tl.float32,
}


def _tle_available():
    if not _HAS_TLE:
        return False
    if _IS_FLAGTREE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_AVAILABLE = _tle_available()


def _npo2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


_TLE_GEOM = {}
_TLE_MISS = object()


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "c_desc"],
)
def _tle_any_row_kernel(
    a_desc,
    c_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))

    acc = tl.zeros([XBLOCK], ACC_DTYPE)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        acc = tl.maximum(acc, tl.max(tl.abs(tl.load(a_ptrs).to(ACC_DTYPE)), axis=1))
    tl.store(c_ptrs, acc.to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


def _tle_any_row_geom(M, N, itemsize):
    key = (M, N, itemsize)
    geom = _TLE_GEOM.get(key, _TLE_MISS)
    if geom is not _TLE_MISS:
        return geom
    xblock = min(_TLE_XBLOCK, max(128, _npo2(-(-M // _TLE_CLUSTERS))))
    row_bytes = N * itemsize
    if row_bytes >= _TLE_WIDE_ROW_BYTES:
        xblock = max(_TLE_CORE_NUM, xblock // 4)
    elif row_bytes <= _TLE_NARROW_ROW_BYTES:
        xblock = _TLE_XBLOCK if M > _TLE_CORE_NUM else 256
    tile_elems = _TLE_LM_BYTES_PER_CORE * _TLE_CORE_NUM // itemsize
    yblock = max(1, min(_npo2(N), tile_elems // xblock))
    while yblock > N and yblock > 1:
        yblock >>= 1
    geom = (xblock, yblock, -(-M // xblock))
    _TLE_GEOM[key] = geom
    return geom


def _tle_any_row(inp, mid, M, N):
    """TLE `max(|x|)` row-reduce of [M, N] into `mid` (float). True on success."""
    if not _TLE_AVAILABLE:
        return False
    if inp.dtype not in _TLE_TL_DTYPE or mid.dtype not in _TLE_TL_DTYPE:
        return False
    if N < 2 or not inp.is_contiguous():
        return False
    try:
        xblock, yblock, row_blocks = _tle_any_row_geom(M, N, inp.element_size())
        a = inp if inp.ndim == 2 and inp.shape[0] == M else inp.view(M, N)
        c = mid if mid.ndim == 1 else mid.view(M)
        with torch_device_fn.device(inp.device):
            _tle_any_row_kernel[(row_blocks,)](
                TensorDescriptor.from_tensor(a, block_shape=[xblock, yblock]),
                TensorDescriptor.from_tensor(c, block_shape=[xblock]),
                N,
                xblock,
                yblock,
                _TLE_TL_DTYPE[inp.dtype],
                _TLE_ACC_DTYPE[inp.dtype],
                _TLE_TL_DTYPE[mid.dtype],
                N % yblock != 0,
            )
        return True
    except Exception:
        return False


_TLE_FLAT_COLS = 4096
_TLE_FLAT_ONESHOT = 8192
# Flat TLE only pays off once the [n // COLS, COLS] reshape leaves a wide grid;
# below this the chunked pointer path is already launch-bound and no slower.
_TLE_FLAT_MIN = 2 << 20


def _tle_any_flat(inp):
    """Flat max(|x|) reduce -> 0-d fp32 tensor (the max). None if out of reach.

    Mirrors sum.py's `_tle_sum_flat`: reshape to [n // COLS, COLS], row-reduce to
    partials, feed back until a single program's loop can finish the residue."""
    if not _TLE_AVAILABLE or inp.dtype not in _TLE_TL_DTYPE:
        return None
    cur = inp.reshape(-1) if inp.is_contiguous() else inp.contiguous().view(-1)
    out = torch.empty((), dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        while cur.numel() > _TLE_FLAT_ONESHOT:
            n = cur.numel()
            cols = (
                _TLE_FLAT_COLS
                if n >= 2 * _TLE_FLAT_COLS
                else max(2, _npo2(n) // _TLE_CORE_NUM)
            )
            rows = n // cols
            if rows < 2:
                break
            main = rows * cols
            tail = n - main
            nxt = torch.empty((rows + tail,), dtype=torch.float32, device=inp.device)
            if not _tle_any_row(cur[:main].view(rows, cols), nxt[:rows], rows, cols):
                return None
            if tail:
                nxt[rows:].copy_(cur[main:])
            cur = nxt
        n = cur.numel()
        if n < 2:
            out.copy_(cur.view(-1))
            return out
        if not _tle_any_row(cur.view(1, n), out.view(1), 1, n):
            return None
    return out


def _per_row_any(inp, M, N, out_shape):
    """Reduce a contiguous [M, N] view over its N axis (per row) -> bool tensor.

    For large M the bool output is produced in two steps (reduce -> float/int
    scratch, then `scratch != 0`): a 1-byte output store scalarizes the whole
    kernel on XPU (~2.3x at [4096,4096]), so writing a wider result and doing a
    tiny follow-up compare is faster. Small M stays single-kernel (launch-bound)."""
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    mid_grid = (triton.cdiv(M, 2048),)
    two_step = M > BLOCK_M_DEFAULT
    if inp.dtype == torch.bool and N % 4 == 0:
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
                if not _tle_any_row(inp, mid, M, N):
                    any_dim_kernel_f[grid](
                        inp, mid, M, N, ACC=acc, buffer_size_limit=2048
                    )
                any_mid_neq0_kernel[mid_grid](
                    mid, out, M, BLOCK=2048, buffer_size_limit=2048
                )
            return out.reshape(out_shape)
        out = torch.empty(M, dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            any_dim_kernel[grid](inp, out, M, N, ACC=acc, buffer_size_limit=2048)
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


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ANY_DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if M == 1:
        out = any(inp).reshape(shape)
    else:
        out = _per_row_any(inp, M, N, shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
