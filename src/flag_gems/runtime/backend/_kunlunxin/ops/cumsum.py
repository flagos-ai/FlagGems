import logging

import torch
import triton
import triton.language as tl
from torch._prims_common import is_boolean_dtype, is_integer_dtype

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)
device = device.name


_ROW_MAX_N = 1024

_FLAT_TM = {
    2: 8,
    4: 8,
    8: 8,
    16: 8,
    32: 8,
    64: 8,
    128: 8,
    256: 4,
    512: 2,
    1024: 4,
    2048: 2,
}

_TL_DTYPES = {
    torch.float16: tl.float32,
    torch.bfloat16: tl.float32,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.bool: tl.int32,
    torch.uint8: tl.int32,
    torch.int8: tl.int32,
    torch.int16: tl.int32,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint64: tl.uint64,
}


@libentry()
@triton.jit
def cumsum_row_kernel(
    inp_ptr,
    out_ptr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One row per program; single 1D tile inclusive scan. (2D axis=1
    tl.cumsum silently mis-computes on this backend, so per-row 1D tiles.)"""
    pid = ext.program_id(0)
    row_offset = pid * N
    n_offsets = tl.arange(0, TILE_N)
    if NEED_MASK:
        mask = n_offsets < N
        x = tl.load(inp_ptr + row_offset + n_offsets, mask=mask, other=0.0)
    else:
        x = tl.load(inp_ptr + row_offset + n_offsets)
    if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
        x = x.to(tl.float32)
    elif (
        tl.constexpr(x.dtype.is_int64()) or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    r = tl.cumsum(x, axis=0)
    if NEED_MASK:
        tl.store(out_ptr + row_offset + n_offsets, r, mask=mask)
    else:
        tl.store(out_ptr + row_offset + n_offsets, r)


@libentry()
@triton.jit
def cumsum_flat_kernel(
    inp_ptr,
    out_ptr,
    N: tl.constexpr,
    BN: tl.constexpr,
    TM: tl.constexpr,
):
    """TM consecutive rows (N elements each) flattened into one BN = TM*N lane
    1D tile, scanned once, then corrected per row: the global scan at lane i
    of row j equals (sum of all rows before j, a per-row constant b_j) plus the
    in-row inclusive scan, so r -= b_j on the j-th band of N lanes recovers
    the exact per-row scan.

    Memory access is fully in-bounds and unmasked (program p owns the
    contiguous block [p*BN, (p+1)*BN) and only ever writes back the addresses
    it loaded, so in-place use is safe), which lets the single 1D tl.cumsum --
    the only scan path this backend lowers correctly -- run at BN >= 1024
    lanes.  Only pure-1D ops are used: any 2D scan / reshape / transpose
    mis-lowers or hits UNREACHABLE on this backend (see _scan_rows_into)."""
    pid = ext.program_id(0)
    base = pid * BN
    idx = tl.arange(0, BN)
    x = tl.load(inp_ptr + base + idx)
    if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
        x = x.to(tl.float32)
    elif (
        tl.constexpr(x.dtype.is_int64()) or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    r = tl.cumsum(x, axis=0)
    for k in tl.static_range(1, TM):
        b = tl.sum(tl.where(idx < k * N, x, 0), axis=0)
        r = r - tl.where((idx >= k * N) & (idx < (k + 1) * N), b, 0)
    tl.store(out_ptr + base + idx, r)


@libentry()
@triton.jit
def cumsum_chunk_scan_kernel(
    inp_ptr,
    out_ptr,
    sums_ptr,
    N,
    C,
    S,
    ACC_DTYPE: tl.constexpr,
    BN: tl.constexpr,
):
    """Unmasked single-shot scan + sum of one full BN-wide chunk of one row
    (BN | N, so every lane is in-bounds by construction).

    Deliberately contains NO runtime loop around the tl.cumsum and NO masked
    load: the previous chunked online scan (carry chained across multiple
    16k-lane scans in one program) corrupted rows past the 12th, and masked
    4096-lane loads mis-compile on this backend (garbage from the OOB/padding
    lanes of the load leaks into the scan and its sum -- reproduced
    deterministically: (64, 16896)/(40, 25600)/(20, 70000) with a partial
    last chunk are wrong, while every exact BN multiple is exact).  Partial
    chunks are therefore scanned on a zero-padded buffer by
    cumsum_chunk_tail_kernel, and the row prefix from all chunks before c is
    applied afterwards by cumsum_chunk_prefix_kernel.

    S is the row stride of `sums` (== C when there is no tail chunk, C+1
    otherwise); the chunk index c below parses with C while the sum goes to
    row * S + c so full-chunk sums and the tail sum (written by the host)
    share one contiguous (M, S) matrix."""
    pid = ext.program_id(0)
    row = pid // C
    c = pid % C
    n_offsets = c * BN + tl.arange(0, BN)
    x = tl.load(inp_ptr + row * N + n_offsets).to(ACC_DTYPE)
    r = tl.cumsum(x, axis=0)
    tl.store(out_ptr + row * N + n_offsets, r)
    tl.store(sums_ptr + row * S + c, tl.sum(x, axis=0))


@libentry()
@triton.jit
def cumsum_chunk_tail_kernel(
    inp_ptr,
    out_ptr,
    ACC_DTYPE: tl.constexpr,
    TL: tl.constexpr,
):
    """Scan of one zero-padded tail: exactly TL lanes, all in-bounds (the
    buffer is TL wide, padding lanes hold 0), hence unmasked.  No reduction
    inside: a tl.sum at 512-2048 lanes (the tail widths) mis-compiles on this
    backend while the scan itself is exact, so the chunk sum is read by the
    host as the scanned tail's last valid element (r[T-1] == sum(all T))."""
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, TL)
    x = tl.load(inp_ptr + pid * TL + n_offsets).to(ACC_DTYPE)
    r = tl.cumsum(x, axis=0)
    tl.store(out_ptr + pid * TL + n_offsets, r)


@libentry()
@triton.jit
def cumsum_chunk_prefix_kernel(
    out_ptr,
    sums_ptr,
    N,
    CF,
    S,
    BN: tl.constexpr,
):
    """Add the scanned prefix of all chunks before chunk c (a per-row
    constant) to every element of full chunk c.  Pure elementwise
    read-modify-write: no reduction, no scan, so no shared-memory/barrier
    interaction; unmasked (c < N // BN, all lanes in-bounds).  The grid is
    (M * CF,): CF full chunks per row, rows of `sums` are S wide (S == CF
    without a tail, CF + 1 with one)."""
    pid = ext.program_id(0)
    row = pid // CF
    c = pid % CF
    prefix = tl.load(sums_ptr + row * S + tl.maximum(c - 1, 0))
    prefix = tl.where(c > 0, prefix, 0)
    n_offsets = c * BN + tl.arange(0, BN)
    x = tl.load(out_ptr + row * N + n_offsets)
    x = x.to(prefix.dtype)
    tl.store(out_ptr + row * N + n_offsets, x + prefix)


@libentry()
@triton.jit
def cumsum_identity_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """N == 1: cumsum along a length-1 row is the elementwise identity
    (out dtype may differ, e.g. int -> int64, so a plain kernel is used)."""
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(inp_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def _scan_rows_into(inp, out, M, N, allow_mm=True):
    """K == 1 (row scan) fast paths: identity for N == 1, prefix-sum matmul when
    _mm_plan accepts the shape (see cumsum_mm_rowblock_kernel), flat multi-row
    sweep for power-of-two N <= 512 (see cumsum_flat_kernel), single-shot row
    kernel for N <= 4096, chunked online scan for N > 4096.

    `allow_mm` is cleared for the matmul path's own remainder rows so the two
    cannot bounce back and forth."""
    f16 = inp.dtype is torch.float16 and out.dtype is torch.float16
    mm_plan = (
        _mm_plan(M, N, f16)
        if allow_mm and inp.dtype in _MM_DTYPES and out.dtype in _MM_DTYPES
        else None
    )
    if N == 1:
        n_elements = inp.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK), 1, 1)
        cumsum_identity_kernel[grid](
            inp,
            out,
            n_elements,
            BLOCK=BLOCK,
            num_warps=4,
            buffer_size_limit=2048,
        )
    elif mm_plan is not None:
        _scan_rows_mm_into(inp, out, M, N, *mm_plan)
    elif N in _FLAT_TM and _FLAT_TM[N] * N >= 64 and M >= _FLAT_TM[N]:
        tm = _FLAT_TM[N]
        n_full = (M // tm) * tm
        BN = tm * N
        grid = (n_full // tm, 1, 1)
        cumsum_flat_kernel[grid](
            inp,
            out,
            N=N,
            BN=BN,
            TM=tm,
            num_warps=8,
            buffer_size_limit=2048,
        )
        if n_full < M:
            inp = inp.reshape(-1).narrow(0, n_full * N, (M - n_full) * N)
            out = out.reshape(-1).narrow(0, n_full * N, (M - n_full) * N)
            M = M - n_full
            _scan_rows_into(inp, out, M, N)
    elif N <= _ROW_MAX_N:
        TILE_N = max(64, triton.next_power_of_2(N))
        need_mask = 1 if TILE_N != N else 0
        num_warps = 8 if TILE_N > 2048 else 4
        grid = (M, 1, 1)
        cumsum_row_kernel[grid](
            inp,
            out,
            N=N,
            TILE_N=TILE_N,
            NEED_MASK=need_mask,
            num_warps=num_warps,
            buffer_size_limit=2048,
        )
    else:
        _scan_rows_group_into(inp, out, M, N)


_GROUP = 1024


@libentry()
@triton.jit
def scan_group_sum_kernel(
    inp, sums, N: tl.constexpr, NG: tl.constexpr, GROUP: tl.constexpr
):
    """(R, NG) grid: sum one GROUP-wide group.  The input is zero-padded to
    exactly NG * GROUP columns, so the load is always fully in-bounds and
    needs no mask (masked loads that can read adjacent memory are the one
    construct this backend mis-compiles)."""
    pid_r = ext.program_id(0)
    pid_g = ext.program_id(1)
    offs = pid_g * GROUP + tl.arange(0, GROUP)
    x = tl.load(inp + pid_r * N + offs)
    if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
        x = x.to(tl.float32)
    elif (
        tl.constexpr(x.dtype.is_int64()) or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    tl.store(sums + pid_r * NG + pid_g, tl.sum(x, axis=0))


@libentry()
@triton.jit
def scan_group_add_kernel(
    inp, out, sums, N: tl.constexpr, NG: tl.constexpr, GROUP: tl.constexpr
):
    """(R, NG) grid: single-shot 1D scan of one group plus the prefix of all
    groups before it (read directly from the pre-scanned `sums`)."""
    pid_r = ext.program_id(0)
    pid_g = ext.program_id(1)
    offs = pid_g * GROUP + tl.arange(0, GROUP)
    x = tl.load(inp + pid_r * N + offs)
    if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
        x = x.to(tl.float32)
    elif (
        tl.constexpr(x.dtype.is_int64()) or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    base = tl.load(sums + pid_r * NG + tl.maximum(pid_g - 1, 0))
    base = base * (pid_g > 0).to(base.dtype)
    r = tl.cumsum(x, axis=0) + base
    tl.store(out + pid_r * N + offs, r)


def _scan_rows_group_into(inp, out, M, N):
    """K == 1, N > _ROW_MAX_N: two-stage group scan.

    Replaces the former chunk-scan + prefix-add path (``cumsum_chunk_*``).
    That path made two full-tensor passes over the data -- one scan+sum pass
    writing ``out`` and ``sums``, then a second read-modify-write prefix-add
    pass over ``out`` -- for ~4N memory traffic.  Here ``scan_group_add_kernel``
    folds the per-group prefix (read directly from the pre-scanned ``sums``)
    into the single-shot group scan, so the data is touched once to sum
    (group_sum: read N) and once to scan+add (group_add: read N, write N) for
    ~3N traffic.  When N is an exact multiple of GROUP there is no partial
    group, so the scan runs straight on ``inp``/``out`` with no zero-pad buffer
    (the common benchmark case 1024x65536 = 16*4096 hits this).  Non-multiples
    fall back to the zero-padded buffer that _scan_mid_into also uses, which
    keeps every load fully in-bounds and unmasked (the one construct this
    backend mis-compiles -- see cumsum_chunk_scan_kernel's note) and carries
    the per-group prefix across kernels rather than inside one program (the
    other backend corner).  bf16/fp16 keep their fp32 accumulation (the group
    kernels up-cast then the fp32 result is cast back on store to ``out``)."""
    inp = inp.reshape(M, N)
    out = out.reshape(M, N)
    GROUP = _GROUP
    n_groups = (N + GROUP - 1) // GROUP
    Np = n_groups * GROUP
    sums_dtype = torch.float32 if inp.dtype.is_floating_point else torch.int64
    with torch_device_fn.device(inp.device):
        sums = torch.empty(M, n_groups, dtype=sums_dtype, device=inp.device)
        if Np == N:
            scan_group_sum_kernel[(M, n_groups)](
                inp, sums, N, n_groups, GROUP, num_warps=8, buffer_size_limit=2048
            )
            _scan_rows_into(sums, sums, M, n_groups)
            scan_group_add_kernel[(M, n_groups)](
                inp, out, sums, N, n_groups, GROUP, num_warps=8, buffer_size_limit=2048
            )
        else:
            xp = torch.zeros(M, Np, dtype=inp.dtype, device=inp.device)
            xp[:, :N] = inp
            out_t = torch.empty(M, Np, dtype=out.dtype, device=out.device)
            scan_group_sum_kernel[(M, n_groups)](
                xp, sums, Np, n_groups, GROUP, num_warps=8, buffer_size_limit=2048
            )
            _scan_rows_into(sums, sums, M, n_groups)
            scan_group_add_kernel[(M, n_groups)](
                xp,
                out_t,
                sums,
                Np,
                n_groups,
                GROUP,
                num_warps=8,
                buffer_size_limit=2048,
            )
            out[:, :N] = out_t[:, :N]


# ---------------------------------------------------------------------------
# Prefix sum as a matmul.  tl.dot moves ~1337 GB/s here against ~273 GB/s for a
# tl.cumsum over the same bytes (measured: 64 MB fp32, BLOCK=8192, num_warps and
# buffer_size_limit have no effect on either), so for the row scan the cheapest
# scan is not a scan at all.
# ---------------------------------------------------------------------------

_MM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# A whole row must be at least one BM x G tile for the matmul to be worth its
# launch; below that the scan path wins.
_MM_MIN_ELEMS = 1 << 12
# Row offsets are computed in int32 inside the kernel.
_MM_MAX_ELEMS = (1 << 31) - 1
# Single-group (NG == 1) tile budget in fp32 lanes: BM * G + G * G.  There is no
# such simple bound once the group loop exists -- the running carry costs extra
# and the measured feasible set is not a function of the tile size alone
# (G=256/BM=512 with NG == 1 compiles while G=128/BM=512 with NG == 32 does not,
# although the latter tile is smaller) -- so the loop case uses the (G, BM)
# pairs below, every one of which was verified to compile.
_MM_FLAT_BUDGET = 200_000
# Group-loop case: G=128 with BM=128 is the widest tile that both compiles and
# wins here; BM=256 at the same G compiles but is slower on every shape with a
# grid under ~20.
_MM_LOOP_BUDGET = 32_768
_MM_BM_CAND = (512, 256, 128, 64, 32, 16)
_MM_MIN_GRID = 8
_MM_U_CACHE = {}


def _mm_tri_u(G, dev, dtype=torch.float32):
    """Upper-triangular ones, cached per (width, device, dtype)."""
    key = (G, dev, dtype)
    u = _MM_U_CACHE.get(key)
    if u is None:
        u = torch.triu(torch.ones(G, G, dtype=dtype, device=dev))
        _MM_U_CACHE[key] = u
    return u


def _mm_plan(M, N, f16=False):
    """(G, BM) for the matmul row-block scan, or None to use the scan path.

    G must be a power of two (it is a tl.arange extent) and must divide N
    exactly, because the kernel is deliberately unmasked -- see
    cumsum_mm_rowblock_kernel.

    BM is chosen to keep `M % BM` small before anything else: the leftover rows
    are re-planned at a narrower BM, and a narrow row block over a long row is
    so much slower per element that a 2.7% tail (M=10000 at BM=512) cost more
    than the 0.16% tail (BM=256) saved by the wider block -- measured
    (10000, 65536) 0.54x at BM=512 against 0.61x at BM=256."""
    if N < 64 or M < 16 or M * N < _MM_MIN_ELEMS or M * N > _MM_MAX_ELEMS:
        return None
    if N in (64, 128, 256):
        # One group per row: no carry, so a much wider row block fits.
        G, budget = N, _MM_FLAT_BUDGET
    elif f16 and N % 256 == 0 and M >= 128:
        # The fp16 kernel feeds tl.dot narrow operands and pays off at a wider
        # contraction (measured (4096, 4096) 0.715x at G=256 vs 0.674x at 128).
        return 256, 128
    elif N % 128 == 0:
        G, budget = 128, _MM_LOOP_BUDGET
    elif N % 64 == 0:
        G, budget = 64, _MM_LOOP_BUDGET
    else:
        return None
    cand = [b for b in _MM_BM_CAND if b <= M and b * G + G * G <= budget]
    if not cand:
        return None
    waste_cap = max(M // 64, 16)
    pool = [b for b in cand if M % b <= waste_cap] or cand
    return G, next((b for b in pool if M // b >= _MM_MIN_GRID), pool[0])


@libentry()
@triton.jit
def cumsum_mm_rowblock_kernel(
    inp, out, u_ptr, N, NG, G: tl.constexpr, BM: tl.constexpr
):
    """cumsum(x) == x @ U with U upper-triangular ones, so Y[i, k] is
    sum_{j <= k} X[i, j].  One program owns BM whole rows and walks that row's
    NG = N // G groups in order, keeping the running row total in registers, so
    the data is read once and written once (~2N traffic) with no `sums` matrix
    and no second pass.

    Every load is unmasked: the caller guarantees N % G == 0 and only launches
    over the first (M // BM) * BM rows (the remainder goes back through
    _scan_rows_into), because a masked partial group leaks the OOB lanes'
    garbage into the dot and its sum -- the same backend corner documented on
    cumsum_chunk_scan_kernel (reproduced here on (33, 1000): rel err 0.24-0.53).

    `carry` MUST stay a computed value (tl.sum) and never a loaded one: in a
    kernel that also holds a 2D load and a tl.dot, broadcasting a *loaded*
    per-row scalar across the (BM, G) tile mis-compiles on this backend.  That
    was verified against broadcast_to / + zeros / * full(1) / * (arange >= 0),
    against folding the prefix into lane 0 ahead of the dot, and against both a
    strided and a contiguous (transposed `sums`) prefix load -- all wrong, while
    the same load stored 1D and the same broadcast of a computed value are both
    exact.  It is what rules out the two-pass group variant here."""
    pid = ext.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    kk = tl.arange(0, G)
    jj = tl.arange(0, G)
    u = tl.load(u_ptr + jj[:, None] * G + kk[None, :]).to(tl.float32)
    carry = tl.zeros([BM], dtype=tl.float32)
    for g in range(NG):
        cols = g * G + kk
        x = tl.load(inp + rm[:, None] * N + cols[None, :]).to(tl.float32)
        y = tl.dot(x, u, input_precision="ieee") + carry[:, None]
        tl.store(out + rm[:, None] * N + cols[None, :], y.to(out.dtype.element_ty))
        carry = carry + tl.sum(x, axis=1)


@libentry()
@triton.jit
def cumsum_mm_rowblock_f16_kernel(
    inp, out, u_ptr, N, NG, G: tl.constexpr, BM: tl.constexpr
):
    """fp16 variant of cumsum_mm_rowblock_kernel that feeds tl.dot the fp16
    operands directly instead of up-casting to fp32 first.

    This is numerically equivalent, not a trade: every U entry is exactly 1.0,
    so each product is exact in fp16 and tl.dot still accumulates in fp32 --
    only the matmul unit changes.  Measured against the up-casting kernel it is
    worth 1.12x on (64, 64), 1.24x on (256, 256), 1.09x on (4096, 4096) and
    2.75x on (10000, 256).  bf16 does NOT get the same treatment: its narrow
    dot runs out of uni_sram for every (G, BM) that has a group loop, and where
    it does compile the up-casting kernel is already faster."""
    pid = ext.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    kk = tl.arange(0, G)
    jj = tl.arange(0, G)
    u = tl.load(u_ptr + jj[:, None] * G + kk[None, :])
    carry = tl.zeros([BM], dtype=tl.float32)
    for g in range(NG):
        cols = g * G + kk
        x = tl.load(inp + rm[:, None] * N + cols[None, :])
        y = tl.dot(x, u) + carry[:, None]
        tl.store(out + rm[:, None] * N + cols[None, :], y.to(out.dtype.element_ty))
        carry = carry + tl.sum(x.to(tl.float32), axis=1)


def _scan_rows_mm_into(inp, out, M, N, G, BM):
    """K == 1 row scan through the matmul row-block kernel.

    Rows past the last whole BM block are re-planned (a narrower BM, so the tail
    still goes through the matmul) and only drop to the scan path once they are
    too few to plan at all -- the kernel itself is unmasked.  Routing the tail
    back to the scan path directly is what made (10000, 256) and (10000, 65536)
    look like 0.69x/0.61x instead of 1.12x/0.65x: no power-of-two BM divides
    10000, so the leftover rows were paying the old price for the whole launch.
    """
    n_full = (M // BM) * BM
    f16 = inp.dtype is torch.float16 and out.dtype is torch.float16
    kernel = cumsum_mm_rowblock_f16_kernel if f16 else cumsum_mm_rowblock_kernel
    kernel[(n_full // BM,)](
        inp,
        out,
        _mm_tri_u(G, inp.device, inp.dtype if f16 else torch.float32),
        N,
        N // G,
        G=G,
        BM=BM,
        num_warps=4,
        buffer_size_limit=2048,
    )
    if n_full == M:
        return
    rest = M - n_full
    tail_in = inp.reshape(-1).narrow(0, n_full * N, rest * N)
    tail_out = out.reshape(-1).narrow(0, n_full * N, rest * N)
    plan = _mm_plan(rest, N, f16)
    # A degenerate row block (BM < 64) is only worth it when the tail plan is
    # flat (one group per row).  Walking a long row that way is slower than the
    # scan path -- measured (10000, 65536): a 16-row BM=16 tail dropped the
    # whole launch from 0.61x to 0.58x, while the same 16-row tail on
    # (10000, 256), where it is a single matmul, is worth 1.51x against 1.09x.
    if plan is not None and (plan[1] >= 64 or plan[0] == N):
        _scan_rows_mm_into(tail_in, tail_out, rest, N, *plan)
    else:
        _scan_rows_into(tail_in, tail_out, rest, N, allow_mm=False)


def _scan_mid_into(inp, out, M, N, K):
    """ "(M, N, K) -> (M, K, N) -> padded group scan -> transpose back."""
    R = M * K
    n_groups = (N + _GROUP - 1) // _GROUP
    Np = n_groups * _GROUP
    with torch_device_fn.device(inp.device):
        inp_t = inp.view(M, N, K).permute(0, 2, 1).contiguous()
        xp = torch.zeros(R, Np, dtype=inp.dtype, device=inp.device)
        xp[:, :N] = inp_t.reshape(R, N)
        sums = torch.empty(
            R,
            n_groups,
            dtype=torch.float32 if inp.dtype.is_floating_point else torch.int64,
            device=inp.device,
        )
        out_t = torch.empty(R, Np, dtype=out.dtype, device=out.device)
        scan_group_sum_kernel[(R, n_groups)](
            xp, sums, Np, n_groups, _GROUP, num_warps=8, buffer_size_limit=2048
        )
        _scan_rows_into(sums, sums, R, n_groups)
        scan_group_add_kernel[(R, n_groups)](
            xp, out_t, sums, Np, n_groups, _GROUP, num_warps=8, buffer_size_limit=2048
        )
    src_t = out_t[:, :N].reshape(M, K, N).permute(0, 2, 1)
    try:
        ok = tle_copy(src_t, out.view(M, N, K))
    except Exception:
        ok = False  # e.g. mainline tle lacks UNI_SRAM (same family as the tril fix)
    if not ok:
        torch.ops.aten._copy_from(
            out_t[:, :N].reshape(M, K, N).permute(0, 2, 1), out.view(M, N, K), False
        )


def cumsum_wrapper(inp, dim=1, dtype=None, out=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
            dtype = torch.int64
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    if K == 1:
        with torch_device_fn.device(inp.device):
            _scan_rows_into(inp, out, M, N)
    else:
        _scan_mid_into(inp, out, M, N, K)
    return out


def cumsum(inp, dim=1, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN CUMSUM")
    return cumsum_wrapper(inp, dim, dtype)


def cumsum_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN CUMSUM_OUT")
    return cumsum_wrapper(inp, dim, dtype, out)


@libentry()
@triton.jit(do_not_specialize=["K", "INNER"])
def normed_cumsum_strided_kernel(inp, out, K, INNER, BLOCK: tl.constexpr):
    row = ext.program_id(0)
    outer = row // INNER
    inner = row % INNER
    base = outer * K * INNER + inner
    offsets = tl.arange(0, BLOCK)
    mask = offsets < K
    x = tl.load(inp + base + offsets * INNER, mask=mask, other=0.0)
    if x.dtype.is_fp16() | x.dtype.is_bf16():
        x = x.to(tl.float32)
    total = tl.sum(x, axis=0)
    result = tl.cumsum(x, axis=0) / total
    tl.store(out + base + offsets * INNER, result, mask=mask)


_FUSED_MAX_N = 4096


@libentry()
@triton.jit
def normed_cumsum_fused_kernel(
    inp,
    out,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    TILE_M: tl.constexpr,
):
    """TILE_M consecutive rows per program (TILE_M = 1 -> one row per program);
    every row is a single-shot 1D scan divided by the row total.  MUST be
    launched with n_rows % TILE_M == 0: a runtime per-row guard cannot wrap a
    scan on this backend (ConvertTritonXPUToLLVM rejects tt.scan inside a
    runtime scf.if, PassManager::run failed), so the remainder rows are covered
    by a TILE_M = 1 launch in the wrapper.  Only TILE_N lanes are live at a
    time, so TILE_M amortizes program-launch overhead on the launch-bound
    huge-n_rows / small-N corner (measured [64,512,512]: 32768 one-row
    programs -> 16.9ms with the old 8192-lane kernel)."""
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    for i in tl.static_range(TILE_M):
        row_off = (pid * TILE_M + i) * N
        if NEED_MASK:
            mask = n_offsets < N
            x = tl.load(inp + row_off + n_offsets, mask=mask, other=0.0)
        else:
            x = tl.load(inp + row_off + n_offsets)
        if tl.constexpr(x.dtype.is_bf16()) or tl.constexpr(x.dtype.is_fp16()):
            x = x.to(tl.float32)
        total = tl.sum(x, axis=0)
        r = tl.cumsum(x, axis=0) / total
        if NEED_MASK:
            tl.store(out + row_off + n_offsets, r, mask=mask)
        else:
            tl.store(out + row_off + n_offsets, r)


@libentry()
@triton.jit(do_not_specialize=["K"])
def normed_cumsum_div_kernel(y, K, ACC_DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    """K > _FUSED_MAX_N: in-place division of a scanned row by the row total
    (the scan's last element).  The scan itself is produced by the proven
    _scan_rows_into chunked path; in-place is safe because each program only
    touches its own row and never re-reads what it wrote."""
    row = ext.program_id(0)
    total = tl.load(y + row * K + (K - 1)).to(ACC_DTYPE)
    for start in range(0, K, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(y + row * K + offs, mask=mask).to(ACC_DTYPE)
        tl.store(y + row * K + offs, x / total, mask=mask)


_NORMED_MM_MAX_K = 512
_NORMED_U_CACHE = {}


def _normed_tri_u(TK, dev):
    key = (TK, dev)
    u = _NORMED_U_CACHE.get(key)
    if u is None:
        u = torch.triu(torch.ones(TK, TK, dtype=torch.float32, device=dev))
        _NORMED_U_CACHE[key] = u
    return u


@libentry()
@triton.jit
def normed_cumsum_mm_kernel(inp, out, u_ptr, M, K, TK: tl.constexpr, BM: tl.constexpr):
    """normed_cumsum as a prefix-sum matmul: Y = X @ U with U upper-triangular
    ones gives Y[m,k] = sum_{j<=k} X[m,j] (the cumulative sum), vectorized across
    BM rows by a single tl.dot (ieee -> fp32-exact); normed = Y / row_total. This
    turns the launch-bound many-row / short-K corner (one scan per row -> millions
    of programs) into a few GEMM tiles.
    NOTE: TK is the next MULTIPLE OF 16 >= K, NOT next_pow2 -- an oversized pow2
    contraction with many zero-padded lanes makes tl.dot drop terms on this
    backend (verified: K=100/TK=128 wrong, K=100/TK=112 correct)."""
    pid = ext.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mmask = rm < M
    kk = tl.arange(0, TK)
    kmask = kk < K
    x = tl.load(
        inp + rm[:, None] * K + kk[None, :],
        mask=mmask[:, None] & kmask[None, :],
        other=0.0,
    ).to(tl.float32)
    jj = tl.arange(0, TK)
    u = tl.load(u_ptr + jj[:, None] * TK + kk[None, :]).to(tl.float32)
    y = tl.dot(x, u, input_precision="ieee")
    inv = 1.0 / tl.sum(x, axis=1)
    r = y * inv[:, None]
    tl.store(
        out + rm[:, None] * K + kk[None, :],
        r.to(out.dtype.element_ty),
        mask=mmask[:, None] & kmask[None, :],
    )


def _normed_mm(inp, out, M, K):
    TK = ((K + 15) // 16) * 16
    BM = 64 if K <= 256 else 32
    u = _normed_tri_u(TK, inp.device)
    grid = (triton.cdiv(M, BM),)
    normed_cumsum_mm_kernel[grid](inp, out, u, M, K, TK=TK, BM=BM, num_warps=4)


def normed_cumsum(inp, dim=-1):
    logger.debug("GEMS_KUNLUNXIN NORMED_CUMSUM")
    assert inp.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    dim = dim % inp.ndim
    inp = inp.contiguous()
    if inp.numel() == 0:
        return torch.empty_like(inp)
    K = inp.size(dim)
    # Short-K prefix-sum-as-matmul fast path (see normed_cumsum_mm_kernel): the
    # many-row / short-K shapes are launch-bound as one-scan-per-row; a GEMM over
    # BM rows collapses that. fp64 and K > cap fall through to the scan path.
    use_mm = (1 < K <= _NORMED_MM_MAX_K) and inp.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )
    if inp.stride(dim) != 1:
        M = inp.numel() // K
        inp_t = inp.movedim(dim, -1).contiguous()
        out_t = torch.empty_like(inp_t)
        with torch_device_fn.device(inp.device):
            if use_mm:
                _normed_mm(inp_t, out_t, M, K)
            else:
                _scan_rows_into(inp_t, out_t, M, K)
                out_t = out_t / out_t.narrow(-1, K - 1, 1)
        return out_t.movedim(-1, dim)
    M = inp.numel() // K
    out = torch.empty_like(inp)
    with torch_device_fn.device(inp.device):
        if use_mm:
            _normed_mm(inp, out, M, K)
            return out
        _scan_rows_into(inp, out, M, K)
        return out / out.narrow(-1, K - 1, 1)
