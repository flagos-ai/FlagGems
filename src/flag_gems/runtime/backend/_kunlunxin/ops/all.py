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
    from triton.runtime import driver
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

# The row kernel below asks for `num_stages=2` on its loop, which is the whole
# difference between 38us and 63us on (4096,4096) fp16. That attribute only has an
# effect when `tritonxpu-tle-pipeline` runs, and the pass reads this variable at
# pass-construction time (first compile), so it has to be set before that -- import
# time is early enough. It is a no-op for loops that do not carry the attribute.
# `sum.py` sets the same thing; setdefault keeps this idempotent.
os.environ.setdefault("TRITONXPU_TLE_PIPELINE", "1")

logger = logging.getLogger(__name__)

# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value

# ---- perf design (2026-09-05, measured on XPU dev6) ----
# Same skeleton as any.py (see the design note there): reduce-op cost dominates, so
# ALL is a native fp32 MIN(|x|) reduction; result = (min != 0). For real values,
# min(|x|) == 0  <=>  some element is +/-0  <=>  NOT all(x != 0), so (min != 0) is
# exactly torch.all (NaN -> |NaN|=NaN -> !=0 -> True, matching torch's nonzero rule).
# fp16 accumulates natively (~253GB/s); bf16 tl.minimum promotes to fp32 on XPU.
# The old per-element `val != 0` AND-tree (int1) was ~97GB/s; the int32-word paths
# were not viable for ALL (a nonzero int32 word does not imply all bytes nonzero).

BLOCK_M_DEFAULT = 64
BLOCK_N_DEFAULT = 512


def _acc_dtype(dt):
    return tl.float16 if dt == torch.float16 else tl.float32


# =============================================================================
# tle min row-reduce fast path (2026-09-16)
# =============================================================================
# Row-wise `all` is a `min |x|` reduce, and the pointer kernel reads at the known
# GM->LM read-rate wall (~545 GB/s): on (4096,4096) f32 it measured 119us against
# torch's 40us. `tle.gpu` moves the same tile with the cluster DMA instead.
# Kernel shape matters more than any knob here: a per-iteration cross-lane
# `tl.min(tile, axis=1)` measured 302us on that shape (32 barrier-bound reduces),
# a 2-D accumulator with ONE reduce at the end 96us, against 140us for the pointer
# path in the same window — and (64,64) at 9.6us against 15.4us.
#
# The output goes out as int8 and is viewed back as bool by the caller, so the
# second launch the pointer path needs for large M disappears too. bf16 rides an
# fp16 *view* (its zero is the same 15-bit pattern) because bf16 handled natively
# on this path measured ~7x the fp16 cost; bool rides an int8 view for the same
# reason plus a pre-existing miscompile in the pointer path (see `_per_row_all`).
_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,  # bool rides here, through an int8 view
}
# (M, N, itemsize) -> (xblock, yblock, row_blocks). Sizing rule: xblock*yblock*4B
# (the fp32 accumulator, the bigger resident) <= 128 KB per buffer, two buffers for
# the pipelined loop. 512-row tiles blow the budget ("TLE kernel stack is over the
# local-memory budget"); yblock 128 measured faster than 64.
_TLE_MIN_GEOM = {}
_TLE_MIN_PLANS = {}
_TLE_MIN_FLAT_LAUNCHERS = None
_TLE_MIN_MISS = object()
# (M, N, dtype) keys whose tle compile failed: a failed compile is re-paid on
# every call (the plan cache above only memoizes successes, and on this triton a
# failure leaves no cache entry behind).  See `_per_row_all` for the measurement.
_TLE_MIN_FAILED = set()


def _npo2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _tle_min_available():
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_MIN_AVAILABLE = _tle_min_available()


def _tle_min_flat_launchers():
    global _TLE_MIN_FLAT_LAUNCHERS
    if _TLE_MIN_FLAT_LAUNCHERS is _TLE_MIN_MISS:
        _TLE_MIN_FLAT_LAUNCHERS = (
            getattr(driver.active, "flat_launchers", None) if _HAS_TLE else None
        )
    return _TLE_MIN_FLAT_LAUNCHERS


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "c_desc"],
)
def _tle_min_row_kernel(
    a_desc,
    c_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    NS: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """`c[m] = (min_j |a[m, j]| != 0)` for a contiguous `[M, N]` input, as int8.

    ⚠️ Two deliberate differences from the sum twin (`sum.py::_tle_sum_row_kernel`):
    the running value is a **2-D accumulator** reduced once at the end, and the
    partial-tile fill is **+inf** -- the neutral element of min (filling 0 would
    turn every "all non-zero" row into a false).
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=tl.int8, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))

    inf = float("inf")
    acc = tl.full([XBLOCK, YBLOCK], inf, ACC_DTYPE)
    for coff in tl.range(0, N, YBLOCK, num_stages=NS):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.full([XBLOCK, YBLOCK], inf, IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        acc = tl.minimum(acc, tl.abs(tl.load(a_ptrs)).to(ACC_DTYPE))
    r = tl.min(acc, axis=1)
    tl.store(c_ptrs, (r != 0).to(tl.int8))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


def _tle_min_geom(M, N, itemsize, acc_itemsize):
    key = (M, N, itemsize, acc_itemsize)
    geom = _TLE_MIN_GEOM.get(key)
    if geom is None:
        # Two buffers of <= 128 KB each (software pipelining double-buffers the LM
        # tile), so xblock*yblock*itemsize <= 131072. Measured bests on (4096,4096):
        # fp16/fp32 (512,128)/(512,64) at NS=2 -> 38us/57us; 512 rows blow the
        # budget at 4 B/element.
        row_bytes = N * itemsize
        if row_bytes <= 2048:
            xblock = 512 if M > 64 else 256
        else:
            xblock = min(512, max(128, _npo2(-(-M // 8))))
        per_buf = 131072
        # Charge whichever of the input tile and the 2-D accumulator is wider: the
        # accumulator scales with the element count, so an int8 input carrying an
        # fp32 accumulator tripped uni_sram at the same tile that fp16 fit.
        yblock = max(
            8, min(_npo2(N), per_buf // (xblock * max(itemsize, acc_itemsize)))
        )
        while yblock > N and yblock > 1:
            yblock >>= 1
        geom = (xblock, yblock, -(-M // xblock))
        _TLE_MIN_GEOM[key] = geom
    return geom


def _tle_min_row(inp, M, N):
    """`inp` [M, N] contiguous fp16/fp32/bf16/bool -> bool [M] (`all` along the last axis).

    Two views carry dtypes the LM path is bad at:
      * bool -> int8: its storage bytes are 0/1, so `min != 0` over int8 is the same
        predicate, and the conversion happens on the LM tile instead of through a
        masked GM bool load (that path is broken -- see the gate comment in
        `_per_row_all`).
      * bf16 -> fp16: "is this element +/-0" is a property of the 15 non-sign bits,
        which both formats lay out identically -- a bf16 is zero exactly when its
        fp16 reinterpretation is. No value ever crosses between the formats, so no
        rounding or range issue applies. bf16 on this path measured ~7x the fp16
        cost (550-600us vs 71us on the same tile), while the reinterpretation is
        free."""
    return _tle_min_row_raw(inp, M, N).view(torch.bool)


def _tle_min_row_raw(inp, M, N):
    """Same as `_tle_min_row` but returns the int8 scratch (0/1 per row).

    Two of these chained do a global `all`: row mins, then one min over the row
    mins. Both stages are the same kernel, so both ride the flat-launcher cache."""
    if inp.dtype == torch.bool:
        src = inp.view(torch.int8)
    elif inp.dtype == torch.bfloat16:
        src = inp.view(torch.float16)
    else:
        src = inp
    acc_itemsize = 2 if src.dtype in (torch.float16, torch.int8) else 4
    xblock, yblock, row_blocks = _tle_min_geom(M, N, src.element_size(), acc_itemsize)
    acc_dtype = tl.float16 if src.dtype in (torch.float16, torch.int8) else tl.float32
    consts = (
        xblock,
        yblock,
        _TLE_TL_DTYPE[src.dtype],
        acc_dtype,
        2,  # NS: one prefetch buffer of DMA overlap (38us vs 63us on (4096,4096) fp16)
        N % yblock != 0,
    )
    plan_key = (M, N, src.dtype) + consts
    plan = _TLE_MIN_PLANS.get(plan_key)
    mid = torch.empty(M, dtype=torch.int8, device=inp.device)
    if plan is None:
        # Descriptor ABI: base pointer, then `.shape` (i32) and `.strides` (i64).
        # Shapes copied from sum.py's row plan; a shorter meta list makes the flat
        # launcher replay two operands short (measured the hard way).
        plan = ((row_blocks,), consts, plan_key, (M, N, N, 1), (M, 1, N))
        _TLE_MIN_PLANS[plan_key] = plan
    grid, consts_, key_, a_meta, c_meta = plan

    launchers = _tle_min_flat_launchers()
    if launchers is None:  # triton without the launcher cache: correct, just slower
        _tle_min_row_kernel[grid](
            TensorDescriptor.from_tensor(src, block_shape=[consts_[0], consts_[1]]),
            TensorDescriptor.from_tensor(mid, block_shape=[consts_[0]]),
            N,
            *consts_,
        )
    else:
        launch, stream = launchers.acquire(_tle_min_row_kernel, key_)
        if launch is None:
            kernel = _tle_min_row_kernel[grid](
                TensorDescriptor.from_tensor(src, block_shape=[consts_[0], consts_[1]]),
                TensorDescriptor.from_tensor(mid, block_shape=[consts_[0]]),
                N,
                *consts_,
            )
            launchers.bind(_tle_min_row_kernel, key_, kernel, grid)
        else:
            launch(stream, src.data_ptr(), *a_meta, mid.data_ptr(), *c_meta)
    return mid


def heur_m_block_size(args):
    return min(triton.next_power_of_2(args["M"]), BLOCK_M_DEFAULT)


def heur_n_block_size(args):
    return min(triton.next_power_of_2(args["N"]), BLOCK_N_DEFAULT)


def heur_m_block_size_p(args):
    return min(triton.next_power_of_2(args["P"]), BLOCK_M_DEFAULT)


def heur_n_block_size_p(args):
    return min(triton.next_power_of_2(args["C"]), BLOCK_N_DEFAULT)


def heur_n_block_size_nw(args):
    return min(triton.next_power_of_2(args["NW"]), BLOCK_N_DEFAULT)


@triton.jit
def _min2(a, b):
    return tl.minimum(a, b)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def all_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
):
    """Per-row ALL: reduce each row's |x| min, store (min != 0)."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * N
    outb = out + rows
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], float("inf"), ACC)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        a = tl.load(inb + cols, mask, other=float("inf")).to(ACC)
        acc = tl.minimum(acc, tl.abs(a))
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(outb, r != 0, row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size_nw,
    },
)
@triton.jit
def all_bool_dim_kernel(
    inw,
    out,
    M,
    NW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-row ALL on a bool tensor viewed as int32 words.

    bool bytes are 0x00/0x01, so every word <= 0x01010101 and a word holds all-four
    True iff it equals 0x01010101. min over words == 0x01010101 <=> all elements True.
    (all_dim's official tests use FLOAT_DTYPES only; this covers torch.all(bool).)"""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * NW
    outb = out + rows
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], 0x01010101, tl.int32)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < NW)
        w = tl.load(inb + cols, mask, other=0x01010101)
        acc = tl.minimum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(outb, r == 0x01010101, row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def all_dim_kernel_f(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
):
    """all_dim_kernel variant storing the raw reduced value (float); see any.py -f note."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inp + rows * N
    outb = out + rows
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], float("inf"), ACC)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < N)
        a = tl.load(inb + cols, mask, other=float("inf")).to(ACC)
        acc = tl.minimum(acc, tl.abs(a))
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(outb, r, row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size_nw,
    },
)
@triton.jit
def all_bool_dim_kernel_f(
    inw,
    out,
    M,
    NW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """all_bool_dim_kernel variant storing the raw int32 reduced value; see any.py -f note."""
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * NW
    outb = out + rows
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], 0x01010101, tl.int32)
    for off in range(0, NW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < NW)
        w = tl.load(inb + cols, mask, other=0x01010101)
        acc = tl.minimum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(outb, r, row_mask)


# ---- global (all elements reduced to a single bool): two-stage ----
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
def all_global_s1(
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
    acc = tl.full([BLOCK_M, BLOCK_N], float("inf"), ACC)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < C)
        a = tl.load(inb + cols, mask, other=float("inf")).to(ACC)
        acc = tl.minimum(acc, tl.abs(a))
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(midb, r, row_mask)


@libentry()
@triton.jit
def all_global_s2(mid, out, P, BLOCK: tl.constexpr, ACC: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < P
    a = tl.load(mid + offs, mask=mask, other=0.0).to(ACC)
    r = tl.reduce(a, axis=0, combine_fn=_min2)
    tl.store(out, r != 0)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size_p,
        "BLOCK_N": heur_n_block_size_p,
    },
)
@triton.jit
def all_global_bool_s1(inw, mid, P, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inb = inw + rows * C
    midb = mid + rows
    row_mask = rows < P
    acc = tl.full([BLOCK_M, BLOCK_N], 0x01010101, tl.int32)
    for off in range(0, C, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        mask = row_mask and (cols < C)
        w = tl.load(inb + cols, mask, other=0x01010101)
        acc = tl.minimum(acc, w)
    r = tl.reduce(acc, axis=1, combine_fn=_min2)[:, None]
    tl.store(midb, r, row_mask)


@libentry()
@triton.jit
def all_global_bool_s2(mid, out, P, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < P
    a = tl.load(mid + offs, mask=mask, other=0x01010101)
    r = tl.reduce(a, axis=0, combine_fn=_min2)
    tl.store(out, r == 0x01010101)


def _global_all(inp):
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
            all_global_bool_s1[(triton.cdiv(p, BLOCK_M_DEFAULT), 1)](
                view.reshape(p, c), mid, p, c, buffer_size_limit=2048
            )
            if p == 1:
                return (mid[0] == 0x01010101).reshape([])
            all_global_bool_s2[(1, 1)](
                mid, out, p, triton.next_power_of_2(p), buffer_size_limit=2048
            )
        return out

    p = _pick_chunks(n)
    c = n // p
    acc = _acc_dtype(inp.dtype)
    mid = torch.empty((p,), dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_global_s1[(triton.cdiv(p, BLOCK_M_DEFAULT), 1)](
            inp.reshape(p, c), mid, p, c, ACC=acc, buffer_size_limit=2048
        )
        if p == 1:
            return (mid[0] != 0).reshape([])
        all_global_s2[(1, 1)](
            mid, out, p, triton.next_power_of_2(p), ACC=acc, buffer_size_limit=2048
        )
    return out


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def all_elem_s1(inp, mid, n, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(inp + offs, mask=mask, other=1)
    nz = tl.where(mask, v != 0, True)
    r = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(mid + pid, r)


@libentry()
@triton.jit
def all_elem_s2(mid, out, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(mid + offs, mask=mask, other=1)
    nz = tl.where(mask, v != 0, True)
    r = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, r)


@libentry()
@triton.jit
def all_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Stage-2 global all reduction (shared with isclose.py)."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    val = tl.load(mid + offset, mask=mask, other=1)
    nz = tl.where(mask, val != 0, True)
    result = tl.reduce(nz, axis=0, combine_fn=reduce_all)
    tl.store(out, result)


def all(inp):
    logger.debug("GEMS_KUNLUNXIN ALL")
    # NOTE (2026-09-16): a TLE flat staging path (row-min over [P, C], then one
    # more pass over the P partials) was tried here and parked: its staging shapes
    # hit `TritonXPUUnrollControl` uni_sram overflows, and a failed compile is only
    # paid for *per call* -- the fallback then masked it into an 8 s/call benchmark
    # cell. The row path below is unaffected; see evidence/all-dim-20260916/ §5.
    if inp.is_contiguous():
        return _global_all(inp)
    n = inp.numel()
    out = torch.empty_strided((), (), dtype=torch.bool, device=inp.device)
    block_size = 2048
    mid_size = triton.cdiv(n, block_size)
    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_elem_s1[(mid_size, 1)](inp, mid, n, block_size, buffer_size_limit=2048)
        if mid_size == 1:
            return mid.reshape([])
        all_elem_s2[(1, 1)](
            mid, out, mid_size, triton.next_power_of_2(mid_size), buffer_size_limit=2048
        )
    return out


def _per_row_all(inp, M, N, out_shape):
    """Reduce a contiguous [M, N] view over its N axis (per row) -> bool tensor.

    See any.py `_per_row_any`: large M uses the wide-scratch two-step form to avoid
    the scalarizing 1-byte bool output store; small M stays single-kernel."""
    # All dtypes ride the tle path when they can: fp16/fp32 natively, bool through
    # an int8 view and bf16 through an fp16 view (see `_tle_min_row` for why those
    # are exact).
    #
    # ⚠️ bool must never reach the pointer kernels *as bool*: their float branch
    # mis-loads 1-byte elements as 0 on XPU for many (M, N) -- an all-True bool
    # (100, 33) came back 100/100 rows "not all" (uint8 broke identically;
    # fp16/fp32/bf16 are unaffected; verified 2026-09-17) -- and the tle compile
    # itself trips `uni_sram` for shapes like (64, 257) bool, so that fallback
    # *is* reached on the default config.  The fallback therefore keeps bool on
    # the int32 word path: pad N up to a multiple of 4 with True (the AND-neutral
    # element) and view the bytes as words.
    key = (M, N, inp.dtype)
    src_ok = inp.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.bool)
    src = inp if (inp.is_contiguous() and N >= 2 and src_ok) else None
    if _TLE_MIN_AVAILABLE and src is not None and key not in _TLE_MIN_FAILED:
        try:
            out = _tle_min_row(src, M, N)
            logger.debug("GEMS_KUNLUNXIN ALL_DIM tle min fast path M=%d N=%d", M, N)
            return out.reshape(out_shape)
        except Exception as exc:  # noqa: BLE001 — any gap re-uses the old path
            # A failed compile is paid again on *every* call: (64, 257) bool
            # measured 7.8 s / 3.5 s / 3.5 s across three calls on this tree
            # against 4.4 s then 0.00 s on the base tree (compiled once, cached).
            # The failure is a deterministic property of (M, N, dtype) -- remember
            # it and skip the retry.
            _TLE_MIN_FAILED.add(key)
            logger.debug(
                "GEMS_KUNLUNXIN ALL_DIM tle min fast path unavailable (%s); "
                "falling back to the pointer kernels",
                exc,
            )
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    two_step = M > BLOCK_M_DEFAULT
    if inp.dtype == torch.bool:
        if N % 4 != 0:
            npad = (N + 3) // 4 * 4
            buf = torch.ones((M, npad), dtype=torch.bool, device=inp.device)
            buf[:, :N] = inp.reshape(M, N)
            inp, N = buf, npad
        inw = inp.reshape(-1).view(torch.int32).reshape(M, N // 4)
        if two_step:
            mid = torch.empty(M, dtype=torch.int32, device=inp.device)
            with torch_device_fn.device(inp.device):
                all_bool_dim_kernel_f[grid](inw, mid, M, N // 4, buffer_size_limit=2048)
            return (mid == 0x01010101).reshape(out_shape)
        out = torch.empty(M, dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            all_bool_dim_kernel[grid](inw, out, M, N // 4, buffer_size_limit=2048)
    else:
        acc = _acc_dtype(inp.dtype)
        if two_step:
            mid_dt = torch.float16 if acc == tl.float16 else torch.float32
            mid = torch.empty(M, dtype=mid_dt, device=inp.device)
            with torch_device_fn.device(inp.device):
                all_dim_kernel_f[grid](inp, mid, M, N, ACC=acc, buffer_size_limit=2048)
            return (mid != 0).reshape(out_shape)
        out = torch.empty(M, dtype=torch.bool, device=inp.device)
        with torch_device_fn.device(inp.device):
            all_dim_kernel[grid](inp, out, M, N, ACC=acc, buffer_size_limit=2048)
    return out.reshape(out_shape)


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIM")
    shape = list(inp.shape)
    orig_ndim = inp.ndim

    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * orig_ndim)
        return out

    assert dim >= -orig_ndim and dim < orig_ndim, "Invalid dim"
    dim = dim % orig_ndim
    N = shape[dim]
    inp = dim_compress(inp, dim)
    shape[dim] = 1
    M = inp.numel() // N

    if M == 1:
        out = all(inp).reshape(shape)
    else:
        out = _per_row_all(inp, M, N, shape)

    if not keepdim and out.ndim > 0:
        out = out.squeeze(dim) if dim < out.ndim else out
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN ALL_DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    orig_ndim = inp.ndim
    assert ((i >= -orig_ndim and i < orig_ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % orig_ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if M == 1:
        out = all(inp).reshape(shape)
    else:
        out = _per_row_all(inp, M, N, shape)

    if not keepdim:
        for d in sorted(dim, reverse=True):
            if out.ndim > 0:
                out = out.squeeze(dim=d)
    return out
