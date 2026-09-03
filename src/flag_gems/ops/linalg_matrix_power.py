import logging
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.linalg_lu_factor_ex import linalg_lu_factor_ex as gems_lu_factor_ex
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ===========================================================================
# Platform policy — one table, one resolved profile (_PLATFORM).
#
# This operator runs the same Triton kernels on every CUDA-family backend
# (nvidia, hygon, amd, iluvatar, metax, thead).  All per-vendor deviations
# are listed in the table below and reach the kernels exclusively as
# tl.constexpr launch arguments (never as module reads inside @triton.jit —
# thead's compiler rejects those).  Every quirk is explained here once;
# kernels only point back to this section.
#
# Row fields (one boolean each):
#   accumulate_fp64  LU / triangular-solve substitution sums accumulate in
#                    fp64 (fp32 inputs benefit most from the fp64 rank-1
#                    updates of the inverse chain).  iluvatar (CoreX) has no
#                    fp64 compute path, so it accumulates in fp32.
#   metax_fp64_dot_fix  MetaX's fp64 MMA (maca_mma v2) cannot lower tl.dot
#                    with any dimension < 16, and scrambles the rows of its
#                    fp64 output tile by a fixed 4x4 transpose inside every
#                    16-row group — _metax_fix_fp64_rows undoes it.  No other
#                    backend is affected; on NVIDIA the fp64 dot path is
#                    untouched.
#   small_smem       The HIP family (hygon, amd) and metax expose only 64 KB
#                    of shared memory per block (NVIDIA allows ~99-227 KB).
#                    The fp64-stored TRSM update gemm stages its dot operands
#                    through shared memory (~73.7 KB at the 3-stage default,
#                    over the 64 KB limit), so the pipeline drops to 2 stages
#                    there.  MetaX additionally needs 1 stage and a wider
#                    K_SLICE for its fp64-MLA quirks (see _trsm_solve_2d).
#   grid_sync_fused  The grid spin-barrier kernels (_grid_sync_kernel and the
#                    barrier-based parallel LU for large-M inverses) hang on
#                    thead (PPU / T-Head Zhenwu 真武) — the first fused
#                    multi-CTA launch never returns — so the fused Tier-3 path
#                    (65 <= M <= 256) falls through to the host-side
#                    binary-exponentiation loop there (one barrier-free
#                    mm/bmm launch per exponent bit).  Every other backend
#                    keeps the fused kernel.
#   thead            thead also has a nondeterministic fp64 tl.dot: once
#                    operand magnitudes leave the O(1) range, the same kernel
#                    call sporadically returns garbage (empirically ~1/8 of
#                    the time at (32,32) fp64, ~half the runs at (256,256);
#                    all positive-power tests stay at O(1) magnitudes and pass
#                    deterministically).  Its negative powers therefore run in
#                    df64 (pure fp32 arithmetic, M <= _DF64_MAX_M) or in an
#                    fp64 chain whose every tl.dot is a (16,16,16) tile for
#                    larger M — see the thead routes in the entry point.
#
# NB: resolve from runtime.device.vendor_name (== flag_gems.vendor_name),
# never flag_gems.vendor_name directly — the ops package is imported before
# that attribute is assigned, so referencing it here would be a circular
# import.  support_fp64 stays a runtime capability read from
# flag_gems.runtime.device.support_fp64 at call time.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlatformProfile:
    accumulate_fp64: bool
    metax_fp64_dot_fix: bool
    small_smem: bool
    grid_sync_fused: bool
    thead: bool


# (acc64, metax, smem64k, gridsync, thead) — per-vendor rows.
_VENDOR_POLICIES = {
    "nvidia": (True, False, False, True, False),
    "hygon": (True, False, True, True, False),
    "amd": (True, False, True, True, False),
    "iluvatar": (False, False, False, True, False),
    "metax": (True, True, True, True, False),
    "thead": (True, False, False, False, True),
}
# Vendors not listed (unknowns / aliases) keep the historical defaults — only
# iluvatar and thead ever deviated from them.
_DEFAULT_POLICY = (True, False, False, True, False)


def _resolve_policy(vendor_name):
    """Platform profile for a vendor string.  Pure function of the vendor
    (unit-testable against the legacy per-vendor flag expressions)."""
    return _PlatformProfile(*_VENDOR_POLICIES.get(vendor_name, _DEFAULT_POLICY))


_PLATFORM = _resolve_policy(flag_gems.runtime.device.vendor_name)
# df64 in-register chain limit: thead/no-fp64 backends' negative powers above
# this M take the fp64 16-tile route instead (see the policy table).
_DF64_MAX_M = 64


@triton.jit
def _metax_fix_fp64_rows(
    z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, G: tl.constexpr
):
    """Undo the MetaX fp64 MMA row permutation.

    MetaX's fp64 dot writes its output tile with rows scrambled by a 4x4
    transpose inside every 16-row group (out[i] = z[4*(i%4) + i//4 + 16*(i//16)]).
    The reshape/trans round-trip un-scrambles the rows and forces the MMA
    result into a blocked layout, so it can be stored or fed into the next dot.
    Only invoked when FIX_FP64 is set (MetaX + fp64); other backends never
    enable it.
    """
    z = tl.reshape(z, (G, 4, 4, BLOCK_N))
    z = tl.trans(z, (0, 2, 1, 3))
    z = tl.reshape(z, (BLOCK_M, BLOCK_N))
    return z


# ---------------------------------------------------------------------------
# Threshold: matrices up to this size use the fused Triton kernel.
# M=64 uses BLOCK=64 with a single tl.dot call per matmul — slightly
# slower than cuBLAS for 64×64 (speedup ~0.7x) but still far better than
# the multi-launch fallback (~0.06x).  Above this we fall back to torch.mm.
# ---------------------------------------------------------------------------
TRITON_THRESHOLD = 64  # max M for any Triton path (single-tile or tiled)


# ===========================================================================
# Kernel 1 — single-tile fused power (one tl.dot per matmul step).
# Used for M <= 32 (and 33 <= M <= 64 in fp64); see the dispatch thresholds
# below.  FIX_FP64/G only activate on metax (policy table).
# ===========================================================================


@libentry()
@triton.heuristics(
    values={
        "num_warps": lambda args: (
            4 if args["BLOCK"] <= 16 else 8 if args["BLOCK"] <= 32 else 8
        ),  # 8 warps matches cuBLAS
        "num_stages": lambda args: (2 if args["BLOCK"] <= 32 else 4),
    }
)
@triton.jit(do_not_specialize=["n"])
def _single_tile_kernel(
    A_ptr,
    out_ptr,
    M,
    n,
    batch_stride,
    BLOCK: tl.constexpr,
    FIX_FP64: tl.constexpr = False,
    G: tl.constexpr = 1,
):
    """One program per batch element.  M <= BLOCK, one tl.dot per matmul."""
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < M)

    a_base = A_ptr + pid * batch_stride
    out_base = out_ptr + pid * batch_stride

    a = tl.load(a_base + offs_m[:, None] * M + offs_n[None, :], mask=mask, other=0.0)

    # fp64 inputs accumulate in fp64 (tl.dot supports fp64 via FMA); fp32/half
    # accumulate in fp32 — matches the generic file's fp64-accumulation policy.
    acc_dtype = a.dtype
    if acc_dtype != tl.float64:
        acc_dtype = tl.float32
    z = a.to(acc_dtype)
    result = z
    has_result = False
    n_remaining = n

    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                result = z
                has_result = True
            else:
                result = tl.dot(result, z, allow_tf32=False)
                if FIX_FP64:
                    result = _metax_fix_fp64_rows(result, BLOCK, BLOCK, G)
            if not FIX_FP64:
                result = tl.where(mask, result, 0.0)
        n_remaining >>= 1
        if n_remaining > 0:
            z = tl.dot(z, z, allow_tf32=False)
            if FIX_FP64:
                z = _metax_fix_fp64_rows(z, BLOCK, BLOCK, G)
            if not FIX_FP64:
                z = tl.where(mask, z, 0.0)

    result = result.to(a.dtype)
    tl.store(out_base + offs_m[:, None] * M + offs_n[None, :], result, mask=mask)


# ---------------------------------------------------------------------------
# df64 (double-single) arithmetic — an fp32 (hi, lo) pair with ~48-bit
# mantissa.  Used by the no-fp64 backends and thead routes (policy table).
# ---------------------------------------------------------------------------


@triton.jit
def _df64_dot(ah, al, bh, bl):
    """df64 matrix product via tl.dot: hi = fp32 dot, lo = cross terms.

    Fast (~tensor-core) but the primary dot's fp32 sum rounding is not
    captured, so it keeps ~1e-7 relative error per matmul that amplifies
    through the power chain (a cond-80 (A⁻¹)³ lands ~2e-6 off, over the fp32
    RESOLUTION rtol).  Used only for M > _DF64_MANUAL_MAX where the error-free
    O(M³) manual product below would be too slow, and no strict tolerance is
    checked.
    """
    ch = tl.dot(ah, bh, allow_tf32=False)
    cl = tl.dot(ah, bl, allow_tf32=False) + tl.dot(al, bh, allow_tf32=False)
    return ch, cl


# error-free df64 matmul below this M; tl.dot above.  Instantiated via
# tl.constexpr(...) — plain ``x: tl.constexpr`` module globals are rejected by
# triton 3.5 (thead's compiler) when read from a @jit kernel.
_DF64_MANUAL_MAX = tl.constexpr(32)


@triton.jit
def _df64_matmul(a_h, a_l, b_h, b_l, M, BLOCK: tl.constexpr):
    """Error-free df64 matrix product C = A @ B (M x M <= BLOCK), one program
    per matrix.

    tl.dot rounds its fp32 accumulation and exposes no residual, so a
    tl.dot-based df64 product keeps ~1e-7 relative error per matmul that
    amplifies through the power chain — a cond-80 (A⁻¹)³ lands ~2e-6 off, over
    the fp32 RESOLUTION rtol.  The scalar fma-based df64 ops (_df64_mul_ds /
    _df64_add) are error-free, so this manual O(M³) product reaches df64
    (~1e-14) accuracy for the small M this kernel targets.
    """
    offs = tl.arange(0, BLOCK)
    r_h = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    r_l = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(M):
        # Column k of A and row k of B (mask-reduce extraction, BLOCK-wide —
        # same axes as the LU kernel's column/row extraction).
        a_k_h = tl.sum(tl.where(offs[None, :] == k, a_h, 0.0), axis=1)
        a_k_l = tl.sum(tl.where(offs[None, :] == k, a_l, 0.0), axis=1)
        b_k_h = tl.sum(tl.where(offs[:, None] == k, b_h, 0.0), axis=0)
        b_k_l = tl.sum(tl.where(offs[:, None] == k, b_l, 0.0), axis=0)
        ph, pl = _df64_mul_ds(
            a_k_h[:, None], a_k_l[:, None], b_k_h[None, :], b_k_l[None, :]
        )
        r_h, r_l = _df64_add(r_h, r_l, ph, pl)
    return r_h, r_l


@triton.jit
def _single_tile_kernel_df64(
    A_h_ptr,
    A_l_ptr,
    out_ptr,
    lo_out_ptr,
    M,
    n,
    batch_stride,
    BLOCK: tl.constexpr,
    STORE_LO: tl.constexpr = False,
    SCALE: tl.constexpr = 1.0,
):
    """Binary exponentiation with df64 accumulation (no-fp64 / thead backend).
    One program per batch element; M <= BLOCK, one error-free df64 matmul per
    step.  Stores the fp32 hi part to ``out_ptr``; when STORE_LO is set also
    stores the lo part to ``lo_out_ptr`` (thead recombines hi+lo into fp64).
    ``SCALE`` multiplies the input pair on load — an exact power of two the
    thead entry uses to keep the chain inside fp32 range for large |n|."""
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < M)

    a_h_base = A_h_ptr + pid * batch_stride
    a_l_base = A_l_ptr + pid * batch_stride
    out_base = out_ptr + pid * batch_stride

    zh = tl.load(a_h_base + offs_m[:, None] * M + offs_n[None, :], mask=mask, other=0.0)
    zl = tl.load(a_l_base + offs_m[:, None] * M + offs_n[None, :], mask=mask, other=0.0)
    zh = zh * SCALE
    zl = zl * SCALE
    rh = zh
    rl = zl
    has_result = False
    n_remaining = n

    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                rh = zh
                rl = zl
                has_result = True
            else:
                if M <= _DF64_MANUAL_MAX:
                    rh, rl = _df64_matmul(rh, rl, zh, zl, M, BLOCK)
                else:
                    rh, rl = _df64_dot(rh, rl, zh, zl)
            rh = tl.where(mask, rh, 0.0)
            rl = tl.where(mask, rl, 0.0)
        n_remaining >>= 1
        if n_remaining > 0:
            if M <= _DF64_MANUAL_MAX:
                zh, zl = _df64_matmul(zh, zl, zh, zl, M, BLOCK)
            else:
                zh, zl = _df64_dot(zh, zl, zh, zl)
            zh = tl.where(mask, zh, 0.0)
            zl = tl.where(mask, zl, 0.0)

    # A power can overflow fp32 for large |n| on a cond-N matrix (e.g. a cond-80
    # A^-31 reaches ~1e59).  The TwoSum/TwoProd renormalisation in the manual
    # df64 ops turns inf operands into NaN; the correct fp32 value there is inf
    # (the fp32-cast reference is inf too), so map NaN -> inf (lo zeroed so a
    # host-side fp64 recombine of hi+lo stays inf).  The tl.dot path already
    # propagates signed inf.
    bad = rh != rh
    rh = tl.where(bad, float("inf"), rh)
    rl = tl.where(bad, 0.0, rl)
    tl.store(out_base + offs_m[:, None] * M + offs_n[None, :], rh, mask=mask)
    if STORE_LO:
        tl.store(
            lo_out_ptr + pid * batch_stride + offs_m[:, None] * M + offs_n[None, :],
            rl,
            mask=mask,
        )


def _matrix_power_df64_pair(A_h, A_l, n, M, shape, scale=1.0):
    """df64 matrix power returning the full (hi, lo) fp32 pair of A^n, where
    A = (A_h, A_l) is the df64 input pair.  Only the single-tile path
    (M <= 64) is supported.  thead recombines ``hi.double() + lo.double()``
    into an fp64 result; the no-fp64 backends keep the hi part only."""
    if len(shape) > 2:
        A_h = A_h.reshape(-1, M, M)
        A_l = A_l.reshape(-1, M, M)
        batch_size = A_h.shape[0]
        out_flat = torch.empty(batch_size, M, M, dtype=torch.float32, device=A_h.device)
        lo_flat = torch.empty(batch_size, M, M, dtype=torch.float32, device=A_h.device)
        batch_stride = M * M
    else:
        A_h = A_h.unsqueeze(0)
        A_l = A_l.unsqueeze(0)
        batch_size = 1
        out_flat = torch.empty(1, M, M, dtype=torch.float32, device=A_h.device)
        lo_flat = torch.empty(1, M, M, dtype=torch.float32, device=A_h.device)
        batch_stride = M * M
    BLOCK = max(triton.next_power_of_2(M), 16)
    _single_tile_kernel_df64[(batch_size,)](
        A_h,
        A_l,
        out_flat,
        lo_flat,
        M,
        n,
        batch_stride,
        BLOCK=BLOCK,
        STORE_LO=True,
        SCALE=scale,
    )
    if len(shape) > 2:
        return out_flat.reshape(shape), lo_flat.reshape(shape)
    return out_flat.squeeze(0), lo_flat.squeeze(0)


def _matrix_power_df64(A_h, A_l, n, M, shape, out=None):
    """df64 matrix power (no-fp64 backend): A = (A_h, A_l) df64 pair, result
    is the fp32 hi part.  Only the single-tile path (M <= 64) is supported.
    """
    hi, _lo = _matrix_power_df64_pair(A_h, A_l, n, M, shape)
    if out is not None:
        out.copy_(hi)
        return out
    return hi


def _df64_power_scale(Xh, k):
    """Smallest exact power-of-two input scale s (0 when none is needed) such
    that the df64 binary-exponentiation chain of (Xh, Xl)^k stays in fp32
    normal range.

    Every intermediate of binary exponentiation is bounded by
    sigma_max(X)^k: each multiply combines powers a + b <= k and each squaring
    stays at or below the highest power of two <= k, so entry values never
    exceed sigma_max(X)^k, and the fp32 accumulation sums stay below
    64 * sigma_max(X)^k — the 116-bit target leaves headroom for both.  From
    sigma_max(X) <= sqrt(||X||_1 * ||X||_inf) <= max(||X||_1, ||X||_inf), the
    row/column sums of the fp32 hi part bound sigma_max (the lo part is
    < 2^-24 of the hi part, inside the margin)."""
    m = Xh.shape[-1]
    Xc = Xh.detach().reshape(-1, m, m).to(device="cpu").double()
    a = Xc.abs()
    r = max(a.sum(dim=-1).amax().item(), a.sum(dim=-2).amax().item(), 1e-30)
    return max(0, math.ceil(math.log2(r) - 116.0 / k))


# ===========================================================================
# Kernel 2 — Grid-level sync fused  (M > 32, single kernel, multi-SM)
# ===========================================================================

TILE = 32


@libentry()
@triton.jit(do_not_specialize=["n"])
def _grid_sync_kernel(
    A_ptr,
    out_ptr,
    scratch_ptr,
    barrier_ptr,
    M,
    n,
    batch_stride,
    TILE_BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    FIX_FP64: tl.constexpr = False,
    G: tl.constexpr = 1,
):
    """Single-kernel binary exponentiation with grid-level sync.

    Grid: ``(batch_size, TILES, TILES)``.
    Each program owns one TILE×TILE output tile and runs the entire
    binary-exponentiation loop.  Between matmul steps all programs
    synchronise via an atomic barrier in global memory, giving
    multi-SM parallelism while keeping everything in one kernel launch.
    """
    pid_batch = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    offs_m = tl.arange(0, TILE_BLOCK)
    offs_k = tl.arange(0, TILE_BLOCK)
    offs_n = tl.arange(0, TILE_BLOCK)

    # Row / col range for this program's tile.
    rm = pid_i * TILE_BLOCK + offs_m
    rn = pid_j * TILE_BLOCK + offs_n
    mask = (rm[:, None] < M) & (rn[None, :] < M)

    # Base pointers for this batch element.
    a_base = A_ptr + pid_batch * batch_stride
    out_base = out_ptr + pid_batch * batch_stride
    scratch_stride = M * M
    # Each batch element gets its own 4 scratch slots (scratch: (4*batch, M, M)).
    scratch_base = scratch_ptr + pid_batch * 4 * scratch_stride
    barrier_base = barrier_ptr + pid_batch * 64

    # total_progs = total programs per batch element = TILES * TILES.
    # Each batch element has its own barrier slot (barrier_ptr + pid_batch * 64).
    n_total = TILES * TILES

    # -----------------------------------------------------------------
    # Step 0 — copy input A tiles to scratch[0] (z) and scratch[2] (result)
    # -----------------------------------------------------------------
    a_tile = tl.load(
        a_base + rm[:, None] * M + rn[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(
        scratch_base + 0 * scratch_stride + rm[:, None] * M + rn[None, :],
        a_tile,
        mask=mask,
    )
    tl.store(
        scratch_base + 2 * scratch_stride + rm[:, None] * M + rn[None, :],
        a_tile,
        mask=mask,
    )

    # ---- Grid barrier: every program must finish writing its scratch[0]/[2]
    # tile before the first matmul reads tiles owned by other programs. ----
    my_count = tl.atomic_add(barrier_base, 1, sem="release")
    barrier_round = (my_count // n_total) + 1
    target = barrier_round * n_total
    while tl.atomic_add(barrier_base, 0, sem="acquire") < target:
        pass

    # Ping-pong indices for the scratch buffer (4 slots per batch).
    #   z_buf:   0 or 1   — current power of two
    #   r_buf:   2 or 3   — current result
    z_buf = 0
    r_buf = 2
    has_result = False
    n_remaining = n

    while n_remaining > 0:
        if n_remaining & 1:
            if not has_result:
                # result = current z (scratch[z_buf]).  z may have advanced past
                # the input A (even n), so scratch[2]'s Step-0 copy of A is not
                # the right starting value; copy this program's z tile instead.
                has_result = True
                zval = tl.load(
                    scratch_base
                    + z_buf * scratch_stride
                    + rm[:, None] * M
                    + rn[None, :],
                    mask=mask,
                    other=0.0,
                )
                tl.store(
                    scratch_base + 2 * scratch_stride + rm[:, None] * M + rn[None, :],
                    zval,
                    mask=mask,
                )
                r_buf = 2
            else:
                dst_r = 5 - r_buf
                _compute_tiled_matmul(
                    scratch_base + r_buf * scratch_stride,
                    scratch_base + z_buf * scratch_stride,
                    scratch_base + dst_r * scratch_stride,
                    M,
                    rm,
                    rn,
                    offs_m,
                    offs_k,
                    offs_n,
                    mask,
                    TILE_BLOCK,
                    TILES,
                    FIX_FP64,
                    G,
                )
                r_buf = dst_r
        n_remaining >>= 1
        if n_remaining > 0:
            dst_z = 1 - z_buf
            _compute_tiled_matmul(
                scratch_base + z_buf * scratch_stride,
                scratch_base + z_buf * scratch_stride,
                scratch_base + dst_z * scratch_stride,
                M,
                rm,
                rn,
                offs_m,
                offs_k,
                offs_n,
                mask,
                TILE_BLOCK,
                TILES,
                FIX_FP64,
                G,
            )
            z_buf = dst_z

        # ---- Grid-level barrier (release/acquire semantics) ----
        my_count = tl.atomic_add(barrier_base, 1, sem="release")
        barrier_round = (my_count // n_total) + 1
        target = barrier_round * n_total
        # Spin with acquire semantics for faster visibility
        while tl.atomic_add(barrier_base, 0, sem="acquire") < target:
            pass

    # ---- Store final result ----
    tl.store(
        out_base + rm[:, None] * M + rn[None, :],
        tl.load(
            scratch_base + r_buf * scratch_stride + rm[:, None] * M + rn[None, :],
            mask=mask,
            other=0.0,
        ),
        mask=mask,
    )


@triton.jit
def _compute_tiled_matmul(
    A_base,
    B_base,
    C_base,
    M,
    rm,
    rn,
    offs_m,
    offs_k,
    offs_n,
    mask_c,
    TILE_BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    FIX_FP64: tl.constexpr = False,
    G: tl.constexpr = 1,
):
    """Compute one tile of C = A @ B, storing result to C_base."""
    acc_dtype = A_base.type.element_ty
    if acc_dtype != tl.float64:
        acc_dtype = tl.float32
    acc = tl.zeros((TILE_BLOCK, TILE_BLOCK), dtype=acc_dtype)
    for tk in range(TILES):
        rk = tk * TILE_BLOCK + offs_k
        mask_a = (rm[:, None] < M) & (rk[None, :] < M)
        mask_b = (rk[:, None] < M) & (rn[None, :] < M)
        a_tile = tl.load(
            A_base + rm[:, None] * M + rk[None, :],
            mask=mask_a,
            other=0.0,
        )
        b_tile = tl.load(
            B_base + rk[:, None] * M + rn[None, :],
            mask=mask_b,
            other=0.0,
        )
        acc += tl.dot(a_tile.to(acc_dtype), b_tile.to(acc_dtype), allow_tf32=False)
    if FIX_FP64:
        # acc has been accumulated in the MetaX fp64 MMA layout; converting it
        # to blocked (which also un-scrambles its rows) must happen before the
        # mask is applied, otherwise the mask would corrupt it in MMA layout.
        acc = _metax_fix_fp64_rows(acc, TILE_BLOCK, TILE_BLOCK, G)
    acc = tl.where(mask_c, acc, 0.0)
    tl.store(C_base + rm[:, None] * M + rn[None, :], acc, mask=mask_c)


# ===========================================================================
# Thresholds for dispatch
# ===========================================================================

SINGLE_TILE_MAX = 32  # single-program fused kernel (fastest for M <= 32)
TILED_MAX = 64  # multi-program tiled kernel  (33 <= M <= 64)


# ===========================================================================
# Local LU factorization (self-contained copy from flag_gems.ops.linalg_lu_factor_ex).
# The external LU op computes fp64 in fp32 (fp32-precision LU); this local copy
# keeps the working matrix in the input dtype so fp64 negatives meet the
# strict pytorch tolerances.  Only small matrices (<= _LU_FACTOR_MAX) use the
# fast in-register kernel; larger ones fall back to the external op.
#
# On backends without fp64 (e.g. ascend, iluvatar) the fp64 path is unavailable,
# so the LU / triangular-solve accumulation uses df64 (double-single, ~48-bit
# mantissa) — two fp32 values — to reach fp32-standard precision.  The df64
# primitives are borrowed from the flag_gems gpu_matrix_rank branch.
# ===========================================================================

_LU_FACTOR_MAX = 64


# ---------------------------------------------------------------------------
# df64 (double-single) arithmetic — a value is an (hi, lo) fp32 pair.  Borrowed
# from the flag_gems gpu_matrix_rank branch (linalg_matrix_rank.py).
# ---------------------------------------------------------------------------


@triton.jit
def _df64_add(h1, l1, h2, l2):
    # Error-free addition of two double-single numbers (Knuth TwoSum on the
    # hi parts, lo parts gathered afterwards, then one renormalization).
    # If the hi sum overflows to +/-inf, the renormalization would turn it
    # into NaN (inf - inf); the correct fp32 value is the inf itself, which
    # preserves the overflow sign.  Powers of a cond-N matrix overflow fp32
    # for large |n| (e.g. cond-80 A^-31 reaches ~1e59), and the fp32-cast
    # reference is +/-inf there too.
    s = h1 + h2
    z = s - h1
    e = (h1 - (s - z)) + (h2 - z)
    lo = l1 + l2 + e
    h = s + lo
    e2 = lo - (h - s)
    is_inf = (s == float("inf")) | (s == float("-inf"))
    h = tl.where(is_inf, s, h)
    e2 = tl.where(is_inf, 0.0, e2)
    return h, e2


@triton.jit
def _df64_mul_ds(a_h, a_l, b_h, b_l):
    # Double-single product: TwoProd on the hi parts plus the cross terms.
    # Overflow guard: a product that overflows to +/-inf keeps its sign (the
    # lo part is meaningless there); without this the renormalization turns it
    # into NaN.
    p = a_h * b_h
    e = tl.fma(a_h, b_h, -p) + a_h * b_l + a_l * b_h
    h = p + e
    ll = e - (h - p)
    is_inf = (p == float("inf")) | (p == float("-inf"))
    h = tl.where(is_inf, p, h)
    ll = tl.where(is_inf, 0.0, ll)
    return h, ll


@triton.jit
def _df64_div_ds(a_h, a_l, b_h, b_l):
    # Double-single division: fp32 quotient plus one df64 correction step.
    q1 = a_h / b_h
    p = q1 * b_h
    pe = tl.fma(q1, b_h, -p)
    r_h, r_l = _df64_add(a_h, a_l, -p, -(pe + q1 * b_l))
    q2 = r_h / b_h
    h = q1 + q2
    ll = q2 - (h - q1)
    return h, ll


@triton.jit
def _df64_sqrt_ds(a_h, a_l):
    # Double-single square root: fp32 root plus one Newton/df64 correction.
    x = tl.sqrt(a_h)
    p = x * x
    pe = tl.fma(x, x, -p)
    r_h, r_l = _df64_add(a_h, a_l, -p, -pe)
    corr = r_h / (2.0 * x)
    h = x + corr
    ll = corr - (h - x)
    not_positive = a_h <= 0.0
    h = tl.where(not_positive, 0.0, h)
    ll = tl.where(not_positive, 0.0, ll)
    return h, ll


@libentry()
@triton.jit
def _lu_factor_kernel(
    A,
    LU,
    PIVOTS,
    INFO,
    PERM,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIVOT: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """In-register LU factorization with partial pivoting, one program per
    matrix.  Tracks the first zero/NaN pivot in INFO.  Copied from the
    flag_gems linalg_lu_factor_ex fast kernel; `work` keeps the input dtype
    (no fp32 downcast) so fp64 factorization is fp64-accurate."""
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=mask, other=0.0)
    if USE_FP64:
        # Accumulate the elimination in fp64 regardless of input dtype — the
        # rank-1 updates are the dominant rounding source in the inverse chain.
        work = work.to(tl.float64)
    # Row-permutation index vector, swapped alongside the rows so the final
    # permutation can be applied on GPU (no device→host copy in the solve).
    perm = tl.arange(0, BLOCK_M)

    info_val = 0

    for j_ind in tl.range(0, K):
        if PIVOT:
            # Extract column j_ind for pivot search.
            col_vals = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
            abs_col = tl.abs(col_vals)
            abs_col = tl.where(rows < j_ind, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            # Swap rows j_ind and pivot_row in work, tracking the permutation.
            row_j = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
            row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
            col_mask = cols[None, :] < N
            work = tl.where((rows[:, None] == j_ind) & col_mask, row_p, work)
            work = tl.where((rows[:, None] == pivot_row) & col_mask, row_j, work)
            pj = tl.sum(tl.where(rows == j_ind, perm, 0), axis=0)
            pp = tl.sum(tl.where(rows == pivot_row, perm, 0), axis=0)
            perm = tl.where(rows == j_ind, pp, perm)
            perm = tl.where(rows == pivot_row, pj, perm)
            tl.store(PIVOTS + pid * K + j_ind, pivot_row + 1)

            # After swap, row j_ind == row_p (already extracted) — reuse.
            u_row = row_p

            # Update col_vals in-place: swap elements at j_ind and pivot_row.
            old_j = tl.sum(tl.where(rows == j_ind, col_vals, 0.0), axis=0)
            old_p = tl.sum(tl.where(rows == pivot_row, col_vals, 0.0), axis=0)
            col_vals = tl.where(rows == j_ind, old_p, col_vals)
            col_vals = tl.where(rows == pivot_row, old_j, col_vals)
        else:
            tl.store(PIVOTS + pid * K + j_ind, j_ind + 1)
            col_vals = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
            u_row = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)

        # Pivot is the diagonal element — index into the column vector.
        pivot = tl.sum(tl.where(rows == j_ind, col_vals, 0.0), axis=0)

        # Track first zero/NaN pivot.
        if info_val == 0:
            if pivot == 0.0 or pivot != pivot:
                info_val = j_ind + 1

        # Scale column below diagonal (L factors) and write back.
        scaled_col = tl.where(rows > j_ind, col_vals / pivot, col_vals)
        work = tl.where(
            (rows[:, None] > j_ind) & (cols[None, :] == j_ind),
            scaled_col[:, None],
            work,
        )

        # Rank-1 trailing update: work[j+1:, j+1:] -= scaled_col * u_row.
        update_mask = (rows[:, None] > j_ind) & (cols[None, :] > j_ind)
        work = tl.where(update_mask, work - scaled_col[:, None] * u_row[None, :], work)

    tl.store(LU + offsets, work.to(LU.dtype.element_ty), mask=mask)
    tl.store(PERM + pid * BLOCK_M + rows, perm, mask=rows < M)
    tl.store(INFO + pid, info_val)


@libentry()
@triton.jit
def _lu_factor_kernel_df64(
    A,
    A_L,
    LU,
    LU_L,
    PIVOTS,
    INFO,
    PERM,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIVOT: tl.constexpr,
):
    """In-register LU factorization with partial pivoting using df64
    (double-single) arithmetic — for backends without fp64.  The working
    matrix is a (hi, lo) fp32 pair loaded from ``A`` / ``A_L`` (fp64 inputs are
    split into the pair before the call so the factorization is df64-accurate
    for the true fp64 input; fp32 inputs pass zeros as the low part); the
    rank-1 elimination updates accumulate in df64 (~48-bit mantissa) so the LU
    factors reach fp32-standard precision.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    offsets = pid * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    a = tl.load(A + offsets, mask=mask, other=0.0)
    al = tl.load(A_L + offsets, mask=mask, other=0.0)
    work_h = a
    work_l = al
    perm = tl.arange(0, BLOCK_M)
    info_val = 0

    for j_ind in tl.range(0, K):
        if PIVOT:
            # Extract column j_ind (df64) — the sum selects the one element.
            col_h = tl.sum(tl.where(cols[None, :] == j_ind, work_h, 0.0), axis=1)
            col_l = tl.sum(tl.where(cols[None, :] == j_ind, work_l, 0.0), axis=1)
            abs_col = tl.abs(col_h)
            abs_col = tl.where(rows < j_ind, -1.0, abs_col)
            abs_col = tl.where(rows < M, abs_col, -1.0)
            pivot_val = tl.max(abs_col, axis=0)
            pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_M), axis=0)

            # Swap rows j_ind and pivot_row (hi + lo), tracking the perm.
            row_j_h = tl.sum(tl.where(rows[:, None] == j_ind, work_h, 0.0), axis=0)
            row_p_h = tl.sum(tl.where(rows[:, None] == pivot_row, work_h, 0.0), axis=0)
            row_j_l = tl.sum(tl.where(rows[:, None] == j_ind, work_l, 0.0), axis=0)
            row_p_l = tl.sum(tl.where(rows[:, None] == pivot_row, work_l, 0.0), axis=0)
            col_mask = cols[None, :] < N
            work_h = tl.where(
                (rows[:, None] == j_ind) & col_mask, row_p_h[None, :], work_h
            )
            work_h = tl.where(
                (rows[:, None] == pivot_row) & col_mask, row_j_h[None, :], work_h
            )
            work_l = tl.where(
                (rows[:, None] == j_ind) & col_mask, row_p_l[None, :], work_l
            )
            work_l = tl.where(
                (rows[:, None] == pivot_row) & col_mask, row_j_l[None, :], work_l
            )
            pj = tl.sum(tl.where(rows == j_ind, perm, 0), axis=0)
            pp = tl.sum(tl.where(rows == pivot_row, perm, 0), axis=0)
            perm = tl.where(rows == j_ind, pp, perm)
            perm = tl.where(rows == pivot_row, pj, perm)
            tl.store(PIVOTS + pid * K + j_ind, pivot_row + 1)

            # After swap, row j_ind == row_p (already extracted) — reuse.
            u_row_h = row_p_h
            u_row_l = row_p_l

            # Update col_vals in-place: swap elements at j_ind and pivot_row.
            old_j_h = tl.sum(tl.where(rows == j_ind, col_h, 0.0), axis=0)
            old_p_h = tl.sum(tl.where(rows == pivot_row, col_h, 0.0), axis=0)
            old_j_l = tl.sum(tl.where(rows == j_ind, col_l, 0.0), axis=0)
            old_p_l = tl.sum(tl.where(rows == pivot_row, col_l, 0.0), axis=0)
            col_h = tl.where(rows == j_ind, old_p_h, col_h)
            col_h = tl.where(rows == pivot_row, old_j_h, col_h)
            col_l = tl.where(rows == j_ind, old_p_l, col_l)
            col_l = tl.where(rows == pivot_row, old_j_l, col_l)
        else:
            tl.store(PIVOTS + pid * K + j_ind, j_ind + 1)
            col_h = tl.sum(tl.where(cols[None, :] == j_ind, work_h, 0.0), axis=1)
            col_l = tl.sum(tl.where(cols[None, :] == j_ind, work_l, 0.0), axis=1)
            u_row_h = tl.sum(tl.where(rows[:, None] == j_ind, work_h, 0.0), axis=0)
            u_row_l = tl.sum(tl.where(rows[:, None] == j_ind, work_l, 0.0), axis=0)

        # Pivot is the diagonal element (df64).
        pivot_h = tl.sum(tl.where(rows == j_ind, col_h, 0.0), axis=0)
        pivot_l = tl.sum(tl.where(rows == j_ind, col_l, 0.0), axis=0)

        # Track first zero/NaN pivot.
        if info_val == 0:
            if pivot_h == 0.0 or pivot_h != pivot_h:
                info_val = j_ind + 1

        # Scale column below diagonal (L factors) — df64 division.
        sh, sl = _df64_div_ds(col_h, col_l, pivot_h, pivot_l)
        scaled_h = tl.where(rows > j_ind, sh, col_h)
        scaled_l = tl.where(rows > j_ind, sl, col_l)
        work_h = tl.where(
            (rows[:, None] > j_ind) & (cols[None, :] == j_ind),
            scaled_h[:, None],
            work_h,
        )
        work_l = tl.where(
            (rows[:, None] > j_ind) & (cols[None, :] == j_ind),
            scaled_l[:, None],
            work_l,
        )

        # Rank-1 trailing update: work -= scaled_col * u_row (df64).
        update_mask = (rows[:, None] > j_ind) & (cols[None, :] > j_ind)
        mh, ml = _df64_mul_ds(
            scaled_h[:, None], scaled_l[:, None], u_row_h[None, :], u_row_l[None, :]
        )
        nh, nl = _df64_add(work_h, work_l, -mh, -ml)
        work_h = tl.where(update_mask, nh, work_h)
        work_l = tl.where(update_mask, nl, work_l)

    tl.store(LU + offsets, work_h, mask=mask)
    tl.store(LU_L + offsets, work_l, mask=mask)
    tl.store(PERM + pid * BLOCK_M + rows, perm, mask=rows < M)
    tl.store(INFO + pid, info_val)


@triton.jit
def _ipiv_to_perm_kernel(pivots_ptr, perm_ptr, n, BLOCK: tl.constexpr):
    """Convert LAPACK IPIV pivots to a row-permutation index vector, on device.

    ``perm`` starts as the identity [0, 1, ..., n-1]; for each step ``i`` the
    two entries ``perm[i]`` and ``perm[pivots[i]-1]`` are swapped (IPIV is
    1-based).  Semantics match LAPACK's getrs row-swap sequence and the CPU
    reference ``_pivots_to_perm_gpu``: the solve then uses
    ``P_B[b, i] = B[b, perm[b, i]]``.

    The pivot sequence is inherently sequential (n steps), but the work is tiny
    (two index loads/stores per step) and each batch element is handled by one
    program, so the whole conversion is a single fast kernel with no
    device->host copy.
    """
    pid = tl.program_id(0)
    base = pid * n
    rows = tl.arange(0, BLOCK)
    tl.store(perm_ptr + base + rows, rows, mask=rows < n)
    for i in range(n):
        p = tl.load(pivots_ptr + base + i).to(tl.int32) - 1
        vi = tl.load(perm_ptr + base + i)
        vp = tl.load(perm_ptr + base + p)
        tl.store(perm_ptr + base + i, vp)
        tl.store(perm_ptr + base + p, vi)


def _pivots_to_perm_gpu(pivots, n):
    """Row-permutation index (B[perm] = Pᵀ B) from LAPACK IPIV pivots, on GPU.

    The old implementation copied the pivots to the host and ran a Python
    permutation loop (a device->host sync + ~n CPU tensor ops, ~10 ms for
    n=1024); the kernel above does the same swaps on device in one launch.
    """
    pv = pivots.reshape(-1, n).contiguous()
    bs = pv.shape[0]
    BLOCK = triton.next_power_of_2(n)
    perm = torch.empty(bs, BLOCK, dtype=torch.int32, device=pivots.device)
    _ipiv_to_perm_kernel[(bs,)](pv, perm, n, BLOCK=BLOCK)
    # The kernel swaps int32 indices; callers use them as index tensors, where
    # torch.gather requires int64 (the old CPU path returned int64 too).  The
    # cast is a cheap on-device copy.
    return perm[:, :n].to(torch.int64).contiguous()


@triton.jit
def _two_I_minus_kernel(in_ptr, out_ptr, total, M, BLOCK: tl.constexpr):
    """out = 2I - in, flat elementwise over (batch, M, M).  Used by the Newton
    inverse refinement step X <- X(2I - A X)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # Row/column within the M×M matrix the flat index belongs to — mod M²
    # drops the batch offset, so the identity diagonal is per-matrix.
    i = (offs // M) % M
    j = offs % M
    eye2 = tl.where(i == j, 2.0, 0.0)
    tl.store(out_ptr + offs, eye2 - v, mask=mask)


def _two_I_minus(T):
    """out = 2I - T over the trailing square matrices (broadcast identity)."""
    total = T.numel()
    out = torch.empty_like(T)
    BLOCK = 1024
    _two_I_minus_kernel[(triton.cdiv(total, BLOCK),)](
        T, out, total, T.shape[-1], BLOCK=BLOCK
    )
    return out


def _lu_factor_ex_local(A, use_df64=False, A_l=None):
    """LU factorization returning (LU, LU_L, pivots, info, perm) for
    matrix_power's inverse.  ``perm`` is the row-permutation index (computed on
    GPU by the kernel, so the solve needs no device→host copy); ``LU_L`` is the
    df64 low part (None unless use_df64).

    Small matrices (<= _LU_FACTOR_MAX) use the local kernel — fp64-accurate on
    fp64 backends, df64 (double-single) accumulation otherwise; larger ones
    fall back to the external flag_gems LU op.
    """
    m, n = A.shape[-2], A.shape[-1]
    if m > _LU_FACTOR_MAX or n > _LU_FACTOR_MAX:
        if use_df64:
            # No-fp64 backends keep the external fp32 LU + df64 solve kernels.
            LU, pivots, info = gems_lu_factor_ex(A)
            return LU, None, pivots, info, _pivots_to_perm_gpu(pivots, m)
        # Parallel physical-layout LU: fp64-accurate for fp64 input (the
        # negative-power path upcasts fp32 to fp64), so the inverse needs no
        # Newton refinement.  Pivots are 0-based IPIV -> 1-based for the perm
        # conversion.
        # NB: thead never reaches this branch — its negative powers are routed
        # to df64 (M <= 64) or rejected (M > 64) in the entry point, because
        # this barrier-based LU deadlocks there (grid_sync_fused is off on
        # thead — see the platform-policy table).
        LU, pivots, info = _lu_factor_parallel(A)
        return LU, None, pivots, info, _pivots_to_perm_gpu(pivots + 1, m)
    input_contiguous = A.contiguous()
    batch_shape = input_contiguous.shape[:-2]
    k = min(m, n)
    batch = input_contiguous.numel() // (m * n)
    block_m = triton.next_power_of_2(m)
    lu = torch.empty_like(input_contiguous)
    pivots = torch.empty((*batch_shape, k), device=A.device, dtype=torch.int32)
    info = torch.empty(batch_shape, device=A.device, dtype=torch.int32)
    perm = torch.empty((batch, block_m), device=A.device, dtype=torch.int32)
    with torch_device_fn.device(A.device):
        if use_df64:
            lu_l = torch.empty_like(input_contiguous)
            if A_l is None:
                A_l = torch.zeros_like(input_contiguous)
            _lu_factor_kernel_df64[(batch,)](
                input_contiguous,
                A_l,
                lu,
                lu_l,
                pivots,
                info,
                perm,
                m,
                n,
                k,
                block_m,
                triton.next_power_of_2(n),
                True,
                num_warps=4,
            )
            return lu, lu_l, pivots, info, perm
        _lu_factor_kernel[(batch,)](
            input_contiguous,
            lu,
            pivots,
            info,
            perm,
            m,
            n,
            k,
            block_m,
            triton.next_power_of_2(n),
            True,
            _PLATFORM.accumulate_fp64,
            num_warps=4,
        )
    return lu, None, pivots, info, perm


# ===========================================================================
# Host helpers
# ===========================================================================


@triton.jit
def _metax_fp64_mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    G: tl.constexpr,
):
    """Tiled fp64 C = A @ B for MetaX, grid (tiles, batch).

    The MetaX backend's own mm kernels cannot lower fp64 tl.dot (their
    accumulator is hardwired to fp32, and its fp64 MMA cannot run with any
    dimension < 16), so the M > 256 host binary-exponentiation path uses this
    self-contained kernel instead.  Each program computes one BLOCK_M x BLOCK_N
    output tile; the fp64 dot result's rows are un-scrambled with
    ``_metax_fix_fp64_rows`` before the store.
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_base = A + pid_b * stride_ab
    b_base = B + pid_b * stride_bb
    c_base = C + pid_b * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kr = k0 * BLOCK_K + rk
        a = tl.load(
            a_base + rm[:, None] * stride_am + kr[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (kr[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_base + kr[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(kr[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a.to(tl.float64), b.to(tl.float64), allow_tf32=False)
    acc = _metax_fix_fp64_rows(acc, BLOCK_M, BLOCK_N, G)
    cmask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        c_base + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(C.dtype.element_ty),
        mask=cmask,
    )


def _metax_fp64_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """fp64 matmul for MetaX (self-contained, see ``_metax_fp64_mm_kernel``)."""
    *_, M, K = A.shape
    *_, K2, N = B.shape
    assert K == K2
    A2 = A.reshape(-1, M, K)
    B2 = B.reshape(-1, K, N)
    batch = A2.shape[0]
    C = torch.empty(batch, M, N, dtype=A.dtype, device=A.device)
    # fp64 tiles are 8 bytes/element; 64x32 + 32x64 per K-step keeps shared
    # memory under the MetaX 64KB limit with a single pipeline stage.
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), batch)
    _metax_fp64_mm_kernel[grid](
        A2,
        B2,
        C,
        M,
        N,
        K,
        A2.stride(0),
        A2.stride(1),
        A2.stride(2),
        B2.stride(0),
        B2.stride(1),
        B2.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        G=BLOCK_M // 16,
        num_warps=4,
        num_stages=1,
    )
    return C.reshape(A.shape[:-2] + (M, N))


@triton.jit
def _thead_fp64_mm16_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
):
    """Tiled fp64 C = A @ B whose every tl.dot is capped at a (32,32,32) tile
    (4 warps) — thead's fp64 tl.dot with wider tiles or more warps
    sporadically returns garbage once operand magnitudes grow (see the thead
    policy row; 4-warp dots up to (64,64,64) were bitwise-deterministic in the
    envelope probe).  The thead M > 64 negative-power chain routes every fp64
    matmul through this kernel.  Grid: (output-tiles, batch)."""
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    grid_n = tl.cdiv(N, 32)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * 32 + tl.arange(0, 32)
    rn = pid_n * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)

    a_base = A + pid_b * stride_ab
    b_base = B + pid_b * stride_bb
    c_base = C + pid_b * stride_cb

    acc = tl.zeros((32, 32), dtype=tl.float64)
    for k0 in tl.range(0, K, 32):
        kr = k0 + rk
        a = tl.load(
            a_base + rm[:, None] * stride_am + kr[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (kr[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_base + kr[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(kr[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a.to(tl.float64), b.to(tl.float64), allow_tf32=False)
    cmask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(
        c_base + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(C.dtype.element_ty),
        mask=cmask,
    )


def _thead_fp64_mm16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """fp64 matmul for the thead large-M deterministic path — every tl.dot is
    a (16,16,16) fp64 tile (see ``_thead_fp64_mm16_kernel``)."""
    *_, M, K = A.shape
    *_, K2, N = B.shape
    assert K == K2
    A2 = A.reshape(-1, M, K)
    B2 = B.reshape(-1, K, N)
    batch = A2.shape[0]
    C = torch.empty(batch, M, N, dtype=A.dtype, device=A.device)
    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 32), batch)
    _thead_fp64_mm16_kernel[grid](
        A2,
        B2,
        C,
        M,
        N,
        K,
        A2.stride(0),
        A2.stride(1),
        A2.stride(2),
        B2.stride(0),
        B2.stride(1),
        B2.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        num_warps=4,
    )
    return C.reshape(A.shape[:-2] + (M, N))


def _matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matmul via the flag_gems Triton kernels (mm for 2D, bmm for batched).

    Looked up on the flag_gems namespace at call time so backend-specialized
    kernels are used when a backend overrides them: iluvatar's mm kernel
    accepts a SPLIT_K tune config that the general mm kernel does not, so the
    general ``flag_gems.ops.mm`` import would fail there with a KeyError.

    On MetaX the backend mm kernels cannot lower fp64 tl.dot at all (fp32-only
    accumulator, and the fp64 MMA needs every dimension >= 16), so fp64 matmuls
    use the self-contained tiled kernel above.
    """
    from flag_gems import bmm, mm

    if _PLATFORM.metax_fp64_dot_fix and A.dtype == torch.float64:
        return _metax_fp64_mm(A, B)
    if A.dim() == 2:
        return mm(A, B)
    return bmm(A, B)


# ===========================================================================
# Triangular solves.  The scalar substitution kernels below are the legacy
# (batch, n)-grid row loops kept for the df64 route (linalg_lu_solve); the
# fast path for every other backend is the blocked _trsm_kernel that follows
# (see its header for the per-block phase helpers).
# ===========================================================================


@triton.jit
def forward_substitution_kernel(
    L_ptr,
    B_ptr,
    Y_ptr,
    n,
    k,
    stride_lb,
    stride_ln,
    stride_lk,
    stride_bb,
    stride_bn,
    stride_bk,
    stride_yb,
    stride_yn,
    stride_yk,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    col_idx = tl.program_id(1)

    if col_idx >= n:
        return

    Lp = L_ptr + pid_batch * stride_lb
    Bp = B_ptr + pid_batch * stride_bb
    Yp = Y_ptr + pid_batch * stride_yb

    for i in range(n):
        y_val = tl.load(Bp + i * stride_bn + col_idx * stride_bk)

        # fp64 accumulation for the substitution sums (fp32 inputs benefit
        # most); fp32 accumulation on backends without an fp64 compute path.
        if USE_FP64:
            acc = tl.zeros((), dtype=tl.float64)
        else:
            acc = tl.zeros((), dtype=tl.float32)
        for j in range(i):
            l_ij = tl.load(Lp + i * stride_ln + j * stride_lk)
            y_j = tl.load(Yp + j * stride_yn + col_idx * stride_yk)
            acc += l_ij * y_j

        y_val = (y_val - acc).to(Yp.dtype.element_ty)
        tl.store(Yp + i * stride_yn + col_idx * stride_yk, y_val)


@triton.jit
def backward_substitution_kernel(
    U_ptr,
    Y_ptr,
    X_ptr,
    n,
    k,
    stride_ub,
    stride_un,
    stride_uk,
    stride_yb,
    stride_yn,
    stride_yk,
    stride_xb,
    stride_xn,
    stride_xk,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    col_idx = tl.program_id(1)

    if col_idx >= n:
        return

    Up = U_ptr + pid_batch * stride_ub
    Yp = Y_ptr + pid_batch * stride_yb
    Xp = X_ptr + pid_batch * stride_xb

    for i in range(n - 1, -1, -1):
        # X[i, col_idx] = (Y[i, col_idx] - sum_{j=i+1}^{n-1} U[i,j] * X[j, col_idx]) / U[i,i]
        y_val = tl.load(Yp + i * stride_yn + col_idx * stride_yk)

        # fp64 accumulation for the substitution sums (fp32 inputs benefit
        # most); fp32 accumulation on backends without an fp64 compute path.
        if USE_FP64:
            acc = tl.zeros((), dtype=tl.float64)
        else:
            acc = tl.zeros((), dtype=tl.float32)
        for j in range(i + 1, n):
            u_ij = tl.load(Up + i * stride_un + j * stride_uk)
            x_j = tl.load(Xp + j * stride_xn + col_idx * stride_xk)
            acc += u_ij * x_j

        u_ii = tl.load(Up + i * stride_un + i * stride_uk)
        x_val = ((y_val - acc) / u_ii).to(Xp.dtype.element_ty)
        tl.store(Xp + i * stride_xn + col_idx * stride_xk, x_val)


@triton.jit
def blocked_forward_substitution_kernel(
    L_ptr,
    B_ptr,
    Y_ptr,
    n,
    k,
    stride_ln,
    stride_lk,
    stride_bb,
    stride_bk,
    stride_yb,
    stride_yk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    if m_start >= n or n_start >= k:
        return

    b_block = tl.load(
        B_ptr + m_start * stride_bb + n_start * stride_bk,
        mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < n)
        & (n_start + tl.arange(0, BLOCK_N)[None, :] < k),
        other=0.0,
    )

    y_block = b_block

    tl.store(
        Y_ptr + m_start * stride_yb + n_start * stride_yk,
        y_block,
        mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < n)
        & (n_start + tl.arange(0, BLOCK_N)[None, :] < k),
    )


@triton.jit
def forward_substitution_kernel_df64(
    L_ptr,
    L_l_ptr,
    B_ptr,
    Y_ptr,
    Y_l_ptr,
    n,
    k,
    stride_lb,
    stride_ln,
    stride_lk,
    stride_l2b,
    stride_l2n,
    stride_l2k,
    stride_bb,
    stride_bn,
    stride_bk,
    stride_yb,
    stride_yn,
    stride_yk,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward substitution L y = b with df64 accumulation (no-fp64 backend).
    L is the (hi, lo) df64 factor pair from the LU kernel; writes the df64
    solution (Y_ptr hi, Y_l_ptr lo)."""
    pid_batch = tl.program_id(0)
    col_idx = tl.program_id(1)
    if col_idx >= n:
        return
    Lp = L_ptr + pid_batch * stride_lb
    L2p = L_l_ptr + pid_batch * stride_l2b
    Bp = B_ptr + pid_batch * stride_bb
    Yp = Y_ptr + pid_batch * stride_yb
    Y2p = Y_l_ptr + pid_batch * stride_yb
    for i in range(n):
        y_h = tl.load(Bp + i * stride_bn + col_idx * stride_bk)
        acc_h = 0.0
        acc_l = 0.0
        for j in range(i):
            l_h = tl.load(Lp + i * stride_ln + j * stride_lk)
            l_l = tl.load(L2p + i * stride_l2n + j * stride_l2k)
            y_j = tl.load(Yp + j * stride_yn + col_idx * stride_yk)
            y_j_l = tl.load(Y2p + j * stride_yn + col_idx * stride_yk)
            th, tl2 = _df64_mul_ds(l_h, l_l, y_j, y_j_l)
            acc_h, acc_l = _df64_add(acc_h, acc_l, th, tl2)
        # y_val = y_val - acc
        rh, rl = _df64_add(y_h, 0.0, -acc_h, -acc_l)
        tl.store(Yp + i * stride_yn + col_idx * stride_yk, rh.to(Yp.dtype.element_ty))
        tl.store(Y2p + i * stride_yn + col_idx * stride_yk, rl.to(Yp.dtype.element_ty))


@triton.jit
def backward_substitution_kernel_df64(
    U_ptr,
    U_l_ptr,
    Y_ptr,
    Y_l_ptr,
    X_ptr,
    X_l_ptr,
    n,
    k,
    stride_ub,
    stride_un,
    stride_uk,
    stride_u2b,
    stride_u2n,
    stride_u2k,
    stride_yb,
    stride_yn,
    stride_yk,
    stride_xb,
    stride_xn,
    stride_xk,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward substitution U x = y with df64 accumulation (no-fp64 backend).
    U, Y are the (hi, lo) df64 pairs; writes the df64 solution (X hi/lo)."""
    pid_batch = tl.program_id(0)
    col_idx = tl.program_id(1)
    if col_idx >= n:
        return
    Up = U_ptr + pid_batch * stride_ub
    U2p = U_l_ptr + pid_batch * stride_u2b
    Yp = Y_ptr + pid_batch * stride_yb
    Y2p = Y_l_ptr + pid_batch * stride_yb
    Xp = X_ptr + pid_batch * stride_xb
    X2p = X_l_ptr + pid_batch * stride_xb
    for i in range(n - 1, -1, -1):
        y_h = tl.load(Yp + i * stride_yn + col_idx * stride_yk)
        y_l = tl.load(Y2p + i * stride_yn + col_idx * stride_yk)
        acc_h = 0.0
        acc_l = 0.0
        for j in range(i + 1, n):
            u_h = tl.load(Up + i * stride_un + j * stride_uk)
            u_l = tl.load(U2p + i * stride_u2n + j * stride_u2k)
            x_j = tl.load(Xp + j * stride_xn + col_idx * stride_xk)
            x_j_l = tl.load(X2p + j * stride_xn + col_idx * stride_xk)
            th, tl2 = _df64_mul_ds(u_h, u_l, x_j, x_j_l)
            acc_h, acc_l = _df64_add(acc_h, acc_l, th, tl2)
        u_h = tl.load(Up + i * stride_un + i * stride_uk)
        u_l = tl.load(U2p + i * stride_u2n + i * stride_u2k)
        # x_val = (y_val - acc) / u_ii
        nh, nl = _df64_add(y_h, y_l, -acc_h, -acc_l)
        xh, xl = _df64_div_ds(nh, nl, u_h, u_l)
        tl.store(Xp + i * stride_xn + col_idx * stride_xk, xh.to(Xp.dtype.element_ty))
        tl.store(X2p + i * stride_xn + col_idx * stride_xk, xl.to(Xp.dtype.element_ty))


# ===========================================================================
# Blocked TRSM (_trsm_kernel): one program per K_SLICE-column group of the
# RHS, BLOCK_SIZE-row diagonal blocks solved by serial substitution followed
# by a tl.dot update of the remaining rows.  The per-block phases exist in
# two variants — the register chain (default) and the fenced (16,16,16)-tile
# chain for thead (D16) — implemented as the four @triton.jit helpers above;
# triton inlines helpers before codegen, so each variant compiles exactly as
# if its statements were written inline at the call site.
# ===========================================================================


@triton.jit
def _trsm_solve_d16(
    A_ptr,
    B_ptr,
    INV_ptr,
    pid,
    stride_a_n,
    stride_b_k,
    blk_start,
    blk_end,
    blk_sz,
    a_cols,
    xr,
    col_offs,
    col_mask,
    BLOCK_SIZE: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """D16 solve variant of _trsm_kernel's diagonal-block phase (thead only,
    see the platform-policy table): each already-solved row is re-read from
    global memory and every step is fenced with a debug_barrier — the same
    fenced pattern as _thead_lu_panel_serial."""
    # thead: the register-chain variant below races on this platform
    # (its cross-thread register dependency relies on the tl.sum sync
    # only — empirically nondeterministic here), so the D16 mode
    # re-reads each already-solved row from global memory and fences
    # every row with a debug_barrier — the same fenced pattern as
    # _thead_lu_panel_serial.
    for r_idx in range(blk_sz):
        row = blk_end - 1 - r_idx if UPPER else blk_start + r_idx
        row_rel = row - blk_start

        # Row of A restricted to the block's triangle.
        a_row = tl.load(
            A_ptr + row * stride_a_n + blk_start + a_cols,
            mask=a_cols < blk_sz,
            other=0.0,
        )
        if UPPER:
            a_row = tl.where(a_cols > row_rel, a_row, 0.0)
        else:
            a_row = tl.where(a_cols < row_rel, a_row, 0.0)

        # Already-solved block rows from global memory (unsolved rows
        # are masked out by a_row).
        xs = tl.load(
            B_ptr + (blk_start + xr) * stride_b_k + col_offs[None, :],
            mask=(xr < blk_sz) & col_mask[None, :],
            other=0.0,
        ).to(A_ptr.dtype.element_ty)
        x_sum = tl.sum(a_row[:, None] * xs, axis=0)

        x_vals = (
            tl.load(
                B_ptr + row * stride_b_k + col_offs,
                mask=col_mask,
                other=0.0,
            )
            - x_sum
        )
        if not UNIT:
            inv_d = tl.load(INV_ptr + pid * BLOCK_SIZE + row_rel)
            x_vals *= inv_d
        tl.store(B_ptr + row * stride_b_k + col_offs, x_vals, mask=col_mask)
        # fence this row's store before the next row's xs reload.
        tl.debug_barrier()


@triton.jit
def _trsm_solve_register(
    A_ptr,
    B_ptr,
    INV_ptr,
    pid,
    stride_a_n,
    stride_b_k,
    blk_start,
    blk_end,
    blk_sz,
    a_cols,
    xr,
    col_offs,
    col_mask,
    BLOCK_SIZE: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """Register-chain solve variant of _trsm_kernel's diagonal-block phase:
    the block's X stays in registers (x_all) across the serial row chain and
    is returned for the update phase."""
    # The block's X stays in REGISTERS across the serial row chain: each
    # row solves against the register tile and writes it back in place,
    # so the per-row step needs no global reload of the whole block.
    # The solve value is still stored to B (the update phase and the
    # next block read it), but the register tile is the source for the
    # next row's dot.
    x_all = tl.load(
        B_ptr + (blk_start + xr) * stride_b_k + col_offs[None, :],
        mask=(xr < blk_sz) & col_mask[None, :],
        other=0.0,
    ).to(A_ptr.dtype.element_ty)
    for r_idx in range(blk_sz):
        row = blk_end - 1 - r_idx if UPPER else blk_start + r_idx
        row_rel = row - blk_start

        # Row of A restricted to the block's triangle.
        a_row = tl.load(
            A_ptr + row * stride_a_n + blk_start + a_cols,
            mask=a_cols < blk_sz,
            other=0.0,
        )
        if UPPER:
            a_row = tl.where(a_cols > row_rel, a_row, 0.0)
        else:
            a_row = tl.where(a_cols < row_rel, a_row, 0.0)

        # Dot against the register tile; the a_row mask selects the
        # already solved rows.  The cross-thread reduction synchronises
        # within the program, so no debug_barrier is needed between
        # rows.
        x_sum = tl.sum(a_row[:, None] * x_all, axis=0)

        x_vals = (
            tl.load(
                B_ptr + row * stride_b_k + col_offs,
                mask=col_mask,
                other=0.0,
            )
            - x_sum
        )
        if not UNIT:
            inv_d = tl.load(INV_ptr + pid * BLOCK_SIZE + row_rel)
            x_vals *= inv_d
        x_all = tl.where((xr == row_rel) & col_mask[None, :], x_vals[None, :], x_all)
        tl.store(B_ptr + row * stride_b_k + col_offs, x_vals, mask=col_mask)
    return x_all


@triton.jit
def _trsm_update_d16(
    A_ptr,
    B_ptr,
    stride_a_n,
    stride_b_k,
    blk_start,
    blk_end,
    blk_sz,
    M_REM,
    rem_s,
    bound,
    col_offs,
    col_mask,
):
    """D16 update variant of _trsm_kernel (thead only): every fp64 dot is
    capped at a (16,16,16) tile — wider fp64 dots nondeterministically return
    garbage on this platform (see the thead policy row)."""
    # thead: keep every fp64 dot at (16,16,16) — wider fp64 dots
    # nondeterministically return garbage on this platform (see
    # the thead policy row).  The solved block's X panel was stored to
    # B above and is re-read here in 16-row x 16-col chunks.
    rr16 = tl.arange(0, 16)
    # fence the serial chain's stores before the cross-thread
    # x16 reloads.
    tl.debug_barrier()
    for m_start in range(0, M_REM, 16):
        rm = rem_s + m_start + rr16
        mask_m = rm < bound
        b_base = B_ptr + rm[:, None] * stride_b_k + col_offs[None, :]
        b_curr = tl.load(
            b_base,
            mask=mask_m[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float64)
        for kc in range(0, blk_sz, 16):
            kr = blk_start + kc + rr16
            km = kr < blk_end
            a16 = tl.load(
                A_ptr + rm[:, None] * stride_a_n + kr[None, :],
                mask=mask_m[:, None] & km[None, :],
                other=0.0,
            ).to(tl.float64)
            x16 = tl.load(
                B_ptr + kr[:, None] * stride_b_k + col_offs[None, :],
                mask=km[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float64)
            b_curr -= tl.dot(a16, x16, allow_tf32=False)
        tl.store(
            b_base,
            b_curr.to(B_ptr.dtype.element_ty),
            mask=mask_m[:, None] & col_mask[None, :],
        )


@triton.jit
def _trsm_update_register(
    A_ptr,
    B_ptr,
    stride_a_n,
    stride_b_k,
    blk_start,
    blk_end,
    blk_sz,
    M_REM,
    rem_s,
    bound,
    x_all,
    a_cols,
    rr,
    col_offs,
    col_mask,
    BM: tl.constexpr,
    K_SLICE: tl.constexpr,
    USE_FP64: tl.constexpr,
    FIX_FP64: tl.constexpr,
    G_BM: tl.constexpr,
):
    """Update phase of _trsm_kernel (non-D16): reuses the register tile
    x_all as the X panel and accumulates the gemm per the backend policy
    (fp64 dot, metax row fix, or iluvatar elementwise fp32)."""
    # Reuse the register tile (x_all) as the X panel.
    for m_start in range(0, M_REM, BM):
        rm = rem_s + m_start + rr
        mask_m = rm < bound
        a_sub = tl.load(
            A_ptr + rm[:, None] * stride_a_n + (blk_start + a_cols)[None, :],
            mask=mask_m[:, None] & (a_cols[None, :] < blk_sz),
            other=0.0,
        )
        if K_SLICE < 8:
            acc = tl.sum(a_sub[:, :, None] * x_all[None, :, :], axis=1)
        elif USE_FP64:
            # fp64 accumulation for the update gemm (fp32 operands
            # kept in fp32 storage, accumulate in fp64 for
            # accuracy) — fp64-capable backends.  On MetaX the
            # fp64 MMA scrambles its output rows, so the result is
            # un-scrambled before the store (see
            # _metax_fix_fp64_rows).
            acc = tl.dot(
                a_sub.to(tl.float64),
                x_all.to(tl.float64),
                allow_tf32=False,
            )
            if FIX_FP64:
                acc = _metax_fix_fp64_rows(acc, BM, K_SLICE, G_BM)
            acc = acc.to(A_ptr.dtype.element_ty)
        else:
            # Backends without an fp64 compute path (iluvatar): its
            # tl.dot also rejects non-batch dims < 16 (K_SLICE=8
            # here), so accumulate the update gemm elementwise in
            # fp32.
            acc = tl.sum(a_sub[:, :, None] * x_all[None, :, :], axis=1)
        b_base = B_ptr + rm[:, None] * stride_b_k + col_offs[None, :]
        b_curr = tl.load(b_base, mask=mask_m[:, None] & col_mask[None, :], other=0.0)
        b_curr = b_curr.to(acc.dtype) - acc
        tl.store(b_base, b_curr, mask=mask_m[:, None] & col_mask[None, :])


@libentry()
@triton.jit
def _trsm_kernel(
    A_ptr,
    B_ptr,
    INV_ptr,
    N,
    K,
    stride_a_n,
    stride_b_k,
    BLOCK_SIZE: tl.constexpr,
    K_SLICE: tl.constexpr,
    BM: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
    USE_FP64: tl.constexpr,
    FIX_FP64: tl.constexpr = False,
    G_BM: tl.constexpr = 1,
    D16: tl.constexpr = False,
):
    """Blocked triangular solve A X = B in place (B <- X), one program per
    K_SLICE-column group of the RHS.

    Rows are processed in BLOCK_SIZE blocks.  The diagonal block is solved by
    serial forward/backward substitution (row-by-row, parallel across the
    K_SLICE columns), then the remaining rows are updated with a tl.dot gemm —
    every data dependency stays within a single program, so the kernel is
    barrier-free apart from the D16 (thead) variant's cross-thread fences in
    the per-block helpers above.  This replaces the (batch, n)-grid scalar
    substitution kernels, whose one-program-per-column serial O(n^2) loop was
    the dominant cost of the negative-power path (~77 ms per solve at n=1024
    vs ~1 ms here).
    """
    pid = tl.program_id(0)
    col_start = pid * K_SLICE
    if col_start >= K:
        return

    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    a_cols = tl.arange(0, BLOCK_SIZE)
    x_rows = tl.arange(0, BLOCK_SIZE)
    x_kcols = tl.arange(0, K_SLICE)
    xr = tl.broadcast_to(x_rows[:, None], (BLOCK_SIZE, K_SLICE))
    col_offs = col_start + x_kcols
    col_mask = col_offs < K
    rr = tl.arange(0, BM)

    for block_idx in range(num_blocks):
        if D16:
            # Cross-thread dependency through global memory inside this CTA
            # (the previous block's update stores are read by this block's
            # solve loads); fence each block iteration (see
            # _thead_lu_panel_serial for the same race without the barrier).
            tl.debug_barrier()
        bk = block_idx if not UPPER else num_blocks - 1 - block_idx
        blk_start = bk * BLOCK_SIZE
        blk_end = tl.minimum(blk_start + BLOCK_SIZE, N)
        blk_sz = blk_end - blk_start

        # ═══ Diagonal block: serial substitution over rows ═══
        # Pre-compute diagonal reciprocals (division out of the serial chain).
        if not UNIT:
            diag_vals = tl.load(
                A_ptr + (blk_start + a_cols) * stride_a_n + (blk_start + a_cols),
                mask=a_cols < blk_sz,
                other=1.0,
            )
            tl.store(
                INV_ptr + pid * BLOCK_SIZE + a_cols,
                1.0 / diag_vals,
                mask=a_cols < blk_sz,
            )

        # Serial solve of the diagonal block — two variants of the same
        # substitution, selected by D16 (thead, see the policy table).
        if D16:
            _trsm_solve_d16(
                A_ptr,
                B_ptr,
                INV_ptr,
                pid,
                stride_a_n,
                stride_b_k,
                blk_start,
                blk_end,
                blk_sz,
                a_cols,
                xr,
                col_offs,
                col_mask,
                BLOCK_SIZE,
                UPPER,
                UNIT,
            )
        else:
            x_all = _trsm_solve_register(
                A_ptr,
                B_ptr,
                INV_ptr,
                pid,
                stride_a_n,
                stride_b_k,
                blk_start,
                blk_end,
                blk_sz,
                a_cols,
                xr,
                col_offs,
                col_mask,
                BLOCK_SIZE,
                UPPER,
                UNIT,
            )

        if D16:
            # fence the serial chain's stores before the D16 update re-reads
            # the solved rows from B across threads (see
            # _thead_lu_panel_serial for the same race without the barrier).
            tl.debug_barrier()

        # ═══ Update: B[rest, kslice] -= A[rest, blk] @ X[blk, kslice] ═══
        need_update = tl.where(UPPER, bk > 0, blk_end < N)
        if need_update:
            M_REM = tl.where(UPPER, blk_start, N - blk_end)
            rem_s = tl.where(UPPER, 0, blk_end)
            bound = tl.where(UPPER, blk_start, N)
            if D16:
                _trsm_update_d16(
                    A_ptr,
                    B_ptr,
                    stride_a_n,
                    stride_b_k,
                    blk_start,
                    blk_end,
                    blk_sz,
                    M_REM,
                    rem_s,
                    bound,
                    col_offs,
                    col_mask,
                )
            else:
                _trsm_update_register(
                    A_ptr,
                    B_ptr,
                    stride_a_n,
                    stride_b_k,
                    blk_start,
                    blk_end,
                    blk_sz,
                    M_REM,
                    rem_s,
                    bound,
                    x_all,
                    a_cols,
                    rr,
                    col_offs,
                    col_mask,
                    BM,
                    K_SLICE,
                    USE_FP64,
                    FIX_FP64,
                    G_BM,
                )


def _trsm_solve_2d(A_tri, B, upper: bool, unitriangular: bool, d16: bool = False):
    """Solve a 2-D triangular system A_tri X = B in place, with ``_trsm_kernel``.

    Self-contained in this operator (no dependency on the shared triangular-solve
    op, whose persistent N>512 path also reuses one barrier counter across its
    batch loop and races on batched inputs).

    ``d16`` (thead large-M path): every fp64 update dot is restricted to a
    (16,16,16) tile — the only fp64-dot shape that stays deterministic on
    thead once operand magnitudes grow — so the solve runs with 16-row update
    sub-tiles and a 16-column K_SLICE.
    """
    n = A_tri.shape[0]
    k = B.shape[1]
    # The update gemm runs in fp64 (USE_FP64 is always true off-iluvatar), so on
    # MetaX it always hits the fp64 dot path.  That backend cannot lower the
    # fp64 dot with N < 16 and scrambles its output rows, so it uses a wider
    # K_SLICE and un-scrambles the result — for fp32- and fp64-stored factors
    # alike.  Everywhere else the narrower slice keeps the iluvatar elementwise
    # fallback small and the fp64 dot legal (NVIDIA handles N = 8 fine).
    fix_fp64 = _PLATFORM.metax_fp64_dot_fix
    K_SLICE = 16 if (fix_fp64 or d16) else 8
    BM = 16 if d16 else 128
    # Software-pipeline depth for the update gemm.  num_stages is a pipelining
    # hint only — it never changes the math — but it decides whether the kernel
    # fits the per-block shared memory: at the 3-stage default the fp64-stored
    # factor path needs ~73.7 KB, over the 64 KB limit on small-smem backends.
    if d16:
        num_stages = 2
    elif fix_fp64:
        num_stages = 1
    elif _PLATFORM.small_smem:
        num_stages = 2
    else:
        num_stages = 3
    num_kslices = (k + K_SLICE - 1) // K_SLICE
    if unitriangular:
        # INV_ptr is only dereferenced when UNIT is false — pass B as a dummy.
        inv = B
        unit_flag = True
    else:
        inv = torch.zeros(num_kslices * 32, dtype=A_tri.dtype, device=A_tri.device)
        unit_flag = False
    _trsm_kernel[(num_kslices,)](
        A_tri,
        B,
        inv,
        n,
        k,
        A_tri.stride(0),
        B.stride(0),
        32,
        K_SLICE,
        BM,
        upper,
        unit_flag,
        _PLATFORM.accumulate_fp64,
        fix_fp64,
        8,  # G_BM = BM // 16 = 128 // 16
        D16=d16,
        num_warps=4,
        # fp64 dot operands are staged through shared memory (8 bytes/element).
        # On 64 KB-shared-memory backends the pipeline depth must drop below
        # the 3-stage default so the 128x32 fp64 update tile fits (see the
        # num_stages computation above); NVIDIA keeps the full pipeline.
        num_stages=num_stages,
    )
    return B


# ===========================================================================
# Solve / inverse assembly (host): LU factors + row permutation -> solve
# (linalg_lu_solve), fp64-accumulation Newton refinement of f32 inverses
# (_newton_refine), and the inverse entry point (_inverse, below the
# large-M LU section it dispatches to).
# ===========================================================================


def linalg_lu_solve(
    LU, perm, B, left=True, adjoint=False, out=None, LU_L=None, use_df64=False
):
    n = LU.shape[-1]
    k = B.shape[-1] if B.dim() > 1 else 1
    # The substitution / TRSM kernels read the PACKED LU directly (forward uses
    # the unit lower triangle, backward the upper triangle and diagonal) — no
    # separate L = tril(LU)+I / U = triu(LU) tensors are built.

    # A = P L U → solve L U X = Pᵀ B.  ``perm`` (GPU, from the LU kernel)
    # already encodes the forward IPIV row swaps; a single gather applies it —
    # no device→host copy, no per-row host loop.
    n_perm = B.shape[-2]
    bs = 1
    for d in B.shape[:-2]:
        bs *= d
    if bs == 1:
        perm_idx = perm if perm.dim() == 1 else perm[0]
        P_B = B[perm_idx[:n_perm]]
    else:
        # torch.gather requires an int64 index; the local LU kernel stores its
        # per-matrix permutation as int32, so upcast here (a cheap on-device
        # copy) to keep one gather path for both local and external LU perms.
        P_B = torch.gather(
            B.reshape(-1, n_perm, k),
            1,
            perm[:, :n_perm].to(torch.int64).unsqueeze(-1).expand(-1, n_perm, k),
        ).reshape(B.shape)

    if use_df64 and LU_L is not None:
        if out is None:
            X = torch.empty_like(B)
        else:
            X = out

        def _batch_stride(t):
            """Batch stride of a matrix tensor (0 for a single matrix)."""
            return t.stride(-3) if t.dim() >= 3 else 0

        batch = 1
        for d in LU.shape[:-2]:
            batch *= d

        Y = torch.empty_like(P_B)
        Y_l = torch.empty_like(P_B)
        X_l = torch.empty_like(P_B)
        forward_substitution_kernel_df64[(batch, n)](
            LU,
            LU_L,
            P_B,
            Y,
            Y_l,
            n,
            k,
            _batch_stride(LU),
            LU.stride(-2),
            LU.stride(-1),
            _batch_stride(LU_L),
            LU_L.stride(-2),
            LU_L.stride(-1),
            _batch_stride(P_B),
            P_B.stride(-2),
            P_B.stride(-1),
            _batch_stride(Y),
            Y.stride(-2),
            Y.stride(-1),
            BLOCK_SIZE=128,
        )
        backward_substitution_kernel_df64[(batch, n)](
            LU,
            LU_L,
            Y,
            Y_l,
            X,
            X_l,
            n,
            k,
            _batch_stride(LU),
            LU.stride(-2),
            LU.stride(-1),
            _batch_stride(LU_L),
            LU_L.stride(-2),
            LU_L.stride(-1),
            _batch_stride(Y),
            Y.stride(-2),
            Y.stride(-1),
            _batch_stride(X),
            X.stride(-2),
            X.stride(-1),
            BLOCK_SIZE=128,
        )
        return X, X_l
    else:
        # Batched tiled TRSM.  The old substitution kernels used a (batch, n)
        # grid — one program per RHS column doing a fully serial O(n^2) scalar
        # loop — ~77 ms per solve at n=1024 (the dominant cost of the whole
        # negative power).  The K-slice TRSM kernel is ~100x faster for the
        # same packed-LU input: the forward solve reads the unit lower
        # triangle, the backward solve the upper triangle + diagonal.
        #
        # fp64 path: LU/P_B are fp64 (fp32 negatives are upcast to fp64 before
        # the inverse), so the TRSM accumulates in fp64.  On no-fp64 backends
        # use_df64 selects the df64 kernels above instead.
        # Flatten the batch dimensions (LU/P_B may be 4-D, e.g. (2,3,M,M)) and
        # solve each matrix independently.
        lu_flat = LU.reshape(-1, LU.shape[-2], LU.shape[-1])
        pb_flat = P_B.reshape(-1, P_B.shape[-2], P_B.shape[-1])
        Xb = torch.empty_like(pb_flat)
        for b in range(pb_flat.shape[0]):
            Y = pb_flat[b].clone()
            _trsm_solve_2d(lu_flat[b], Y, upper=False, unitriangular=True)
            _trsm_solve_2d(lu_flat[b], Y, upper=True, unitriangular=False)
            Xb[b] = Y
        X = Xb.reshape(P_B.shape)

    return X


def _newton_refine(A: torch.Tensor, X: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """Improve an approximate inverse X ≈ A⁻¹ via Newton iteration.

    X <- X(2I - A X) converges quadratically to A⁻¹.  For f32 inputs the LU
    factors are f32-stored (~1e-7 backward error), so the TRSM solve is only
    f32-accurate; the refinement accumulates in fp64 (the matmul operands are
    upcast for the dots) to reach the fp32-representable floor in one step.
    fp64 inputs are already accurate and refinement is a no-op by default.

    All arithmetic here is flag_gems Triton matmuls — no torch-native compute
    operators are used.
    """
    m = A.shape[-1]
    is_batched = A.dim() > 2
    A3 = A.reshape(-1, m, m).contiguous()
    X3 = X.reshape(-1, m, m).contiguous()
    # f32 storage + fp64 accumulation: the LU/TRSM run in fp32 (fp32-stored
    # factors), and the refinement upcasts to fp64 so the inverse is fp64-
    # accurate.  Keeping X in fp64 matters: an fp32-stored inverse's ~6e-8
    # error would amplify past the f32 atol once the result is raised to |n|.
    use_f64 = A3.dtype == torch.float32
    A64 = A3.double() if use_f64 else A3
    X64 = X3.double() if use_f64 else X3
    for _ in range(iters):
        tmp = _matmul(A64, X64)  # A X
        g = _two_I_minus(tmp)  # 2I - A X
        X64 = _matmul(X64, g)  # X(2I - A X)
    X64 = X64.reshape(A.shape) if is_batched else X64.squeeze(0)
    return X64


# ===========================================================================
# Parallel physical-layout blocked LU  (partial pivoting, self-contained)
#
# Standard getrf structure: rows are physically swapped during factorization so
# every read is contiguous (no perm-scatter).  The panel factor is parallelised
# across R row-groups with two soft barriers per column; the pivot application
# to the left/trailing columns is done by separate parallel kernels.  The LU is
# fp64-accurate for fp64 input (~1e-14 backward error), so the inverse from it
# needs no Newton refinement.
# ===========================================================================

_LU_PAR_R = 32
_LU_PAR_PANEL = 32
_LU_PAR_TILE_M = 64
_LU_PAR_TILE_N = 128


@triton.jit
def _lu_panel_par(
    LU_ptr,
    pivots_ptr,
    info_ptr,
    sval_ptr,
    srow_ptr,
    spiv_ptr,
    barrier_ptr,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    R: tl.constexpr,
    GROUP: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    """Factor LU[K0:M, K0:K0+PANEL] with partial pivoting.  Grid (R, batch).

    Rows [K0, M) are split into R row-groups.  Per column jj (global j = K0+jj):
    each group contributes its local |max| of column jj -> barrier A -> the
    global pivot row p is reduced from the scratch -> the leader reads the pivot
    value and physically swaps rows j and p (panel columns) -> barrier B -> all
    groups scale / rank-1 update their rows.  Only the leader touches LU rows
    j/p during the swap, and the pivot value is broadcast via a scratch slot, so
    no group ever races on LU[p, j] with the swap.
    """
    pid_g = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = K0 + pid_g * GROUP + tl.arange(0, GROUP)
    rowmask = rows < M
    base = pid_b * M * N
    cols = tl.arange(0, COL_BLOCK)
    colmask = cols < PANEL

    sb = pid_b * (2 * R + 8)
    bA = barrier_ptr + pid_b * 8 + 0
    bB = barrier_ptr + pid_b * 8 + 1
    R_TOT: tl.constexpr = R
    RIDX = tl.arange(0, R)

    info_val = 0
    for jj in tl.range(0, PANEL):
        j = K0 + jj
        # 1. local max over this group's rows r >= j of |LU[r, j]|
        col_vals = tl.load(LU_ptr + base + rows * N + j, mask=rowmask, other=0.0)
        abs_col = tl.where(rows >= j, tl.abs(col_vals), -1.0)
        abs_col = tl.where(rowmask, abs_col, -1.0)
        loc_val = tl.max(abs_col, axis=0)
        loc_row = tl.min(tl.where(abs_col == loc_val, rows, M), axis=0)
        loc_row = tl.where(loc_val < 0.0, M + 1, loc_row)
        tl.store(sval_ptr + sb + pid_g, loc_val)
        tl.store(srow_ptr + sb + pid_g, loc_row)

        # 2. barrier A
        my = tl.atomic_add(bA, 1, sem="release")
        target = (my // R_TOT + 1) * R_TOT
        while tl.atomic_add(bA, 0, sem="acquire") < target:
            pass

        # 3. global pivot p (logical row)
        gm = tl.load(sval_ptr + sb + RIDX)
        gr = tl.load(srow_ptr + sb + RIDX)
        gv = tl.max(gm, axis=0)
        p = tl.min(tl.where(gm == gv, gr, M), axis=0)

        # 4. leader: read pivot value (before the swap), swap rows j and p in
        #    the panel columns, and broadcast pivot_val via a scratch slot.
        if pid_g == 0:
            pivot_val = tl.load(LU_ptr + base + p * N + j)
            tl.store(spiv_ptr + pid_b, pivot_val)
            if info_val == 0:
                if pivot_val == 0.0 or pivot_val != pivot_val:
                    info_val = j + 1
            if j != p:
                rj = tl.load(
                    LU_ptr + base + j * N + (K0 + cols), mask=colmask, other=0.0
                )
                rp = tl.load(
                    LU_ptr + base + p * N + (K0 + cols), mask=colmask, other=0.0
                )
                tl.store(LU_ptr + base + j * N + (K0 + cols), rp, mask=colmask)
                tl.store(LU_ptr + base + p * N + (K0 + cols), rj, mask=colmask)
        tl.store(pivots_ptr + pid_b * M + j, p)

        # 5. barrier B (orders the swap before the update and the next column)
        my = tl.atomic_add(bB, 1, sem="release")
        target = (my // R_TOT + 1) * R_TOT
        while tl.atomic_add(bB, 0, sem="acquire") < target:
            pass

        # 6. scale + rank-1 update for this group's rows (panel columns).
        pivot_val = tl.load(spiv_ptr + pid_b)
        u_row = tl.load(LU_ptr + base + j * N + (K0 + cols), mask=colmask, other=0.0)
        offs = base + rows[:, None] * N + (K0 + cols[None, :])
        mask2 = rowmask[:, None] & colmask[None, :]
        sl = tl.load(LU_ptr + offs, mask=mask2, other=0.0).to(LU_ptr.dtype.element_ty)
        col_vals = tl.sum(tl.where(cols[None, :] == jj, sl, 0.0), axis=1)
        scaled = tl.where(rows > j, col_vals / pivot_val, col_vals)
        sl = tl.where((rows[:, None] > j) & (cols[None, :] == jj), scaled[:, None], sl)
        upd = (rows[:, None] > j) & (cols[None, :] > jj)
        sl = tl.where(upd, sl - scaled[:, None] * u_row[None, :], sl)
        tl.store(LU_ptr + offs, sl, mask=mask2)

    if pid_g == 0:
        tl.store(info_ptr + pid_b, info_val)


@triton.jit
def _lu_apply_left_par(
    LU_ptr,
    pivots_ptr,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply the panel's row swaps to the left columns [0, K0) of rows [K0, M).
    Grid: (col-tiles, batch).  Sequential over the PANEL pivots (with a
    debug_barrier so Triton cannot reorder the chained in-place swaps)."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    colmask = cols < K0
    base = pid_b * M * N
    for jj in tl.range(0, PANEL):
        j = K0 + jj
        p = tl.load(pivots_ptr + pid_b * M + j)
        if j != p:
            rj_off = base + j * N + cols
            rp_off = base + p * N + cols
            rj = tl.load(LU_ptr + rj_off, mask=colmask, other=0.0)
            rp = tl.load(LU_ptr + rp_off, mask=colmask, other=0.0)
            tl.store(LU_ptr + rj_off, rp, mask=colmask)
            tl.store(LU_ptr + rp_off, rj, mask=colmask)
            tl.debug_barrier()


@triton.jit
def _lu_swap_right_solve_par(
    LU_ptr,
    pivots_ptr,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply the panel's row swaps to the trailing columns [K0+PANEL, N) of the
    panel rows [K0, K0+PANEL), then solve the U rows
    L[K0:K0+B] U = A[K0:K0+B, K0+B:N].  Grid: (col-tiles, batch)."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    cols = K0 + PANEL + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    colmask = cols < N
    brows = tl.arange(0, BLOCK_M)
    base = pid_b * M * N

    # swap rows [K0, K0+PANEL) with their pivots in these columns
    for jj in tl.range(0, PANEL):
        j = K0 + jj
        p = tl.load(pivots_ptr + pid_b * M + j)
        if j != p:
            rj_off = base + j * N + cols
            rp_off = base + p * N + cols
            rj = tl.load(LU_ptr + rj_off, mask=colmask, other=0.0)
            rp = tl.load(LU_ptr + rp_off, mask=colmask, other=0.0)
            tl.store(LU_ptr + rj_off, rp, mask=colmask)
            tl.store(LU_ptr + rp_off, rj, mask=colmask)
            tl.debug_barrier()

    # solve U rows
    offs = base + (K0 + brows)[:, None] * N + cols[None, :]
    mask2 = colmask[None, :]
    vals = tl.load(LU_ptr + offs, mask=mask2, other=0.0).to(LU_ptr.dtype.element_ty)
    for jj in tl.range(0, PANEL):
        row_j = tl.sum(tl.where(brows[:, None] == jj, vals, 0.0), axis=0)
        l_off = base + (K0 + brows)[:, None] * N + (K0 + jj)
        l_col = tl.load(LU_ptr + l_off, mask=brows[:, None] < M, other=0.0).to(
            LU_ptr.dtype.element_ty
        )
        l_col = tl.where(brows[:, None] <= jj, 0.0, l_col)
        vals = tl.where(brows[:, None] > jj, vals - l_col * row_j[None, :], vals)
    tl.store(LU_ptr + offs, vals, mask=mask2)


@triton.jit
def _lu_trailing_update_par(
    LU_ptr,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """A[K0+PANEL:M, K0+PANEL:N] -= L[K0+PANEL:M, K0:K0+PANEL] @ U[K0:K0+PANEL, K0+PANEL:N].
    Grid: (row-tiles, col-tiles, batch).  The dot accumulates in fp64 for
    accuracy, matching the rest of the fp64 inverse path."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    rows = K0 + PANEL + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = K0 + PANEL + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bidx = tl.arange(0, BLOCK_B)
    base = pid_b * M * N
    tile_offs = base + rows[:, None] * N + cols[None, :]
    tile_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tile = tl.load(LU_ptr + tile_offs, mask=tile_mask, other=0.0).to(
        LU_ptr.dtype.element_ty
    )
    l_offs = base + rows[:, None] * N + (K0 + bidx[None, :])
    u_offs = base + (K0 + bidx[:, None]) * N + cols[None, :]
    l_mask = (rows[:, None] < M) & (bidx[None, :] < PANEL)
    u_mask = (bidx[:, None] < PANEL) & (cols[None, :] < N)
    l_vals = tl.load(LU_ptr + l_offs, mask=l_mask, other=0.0).to(
        LU_ptr.dtype.element_ty
    )
    u_vals = tl.load(LU_ptr + u_offs, mask=u_mask, other=0.0).to(
        LU_ptr.dtype.element_ty
    )
    update = tl.dot(l_vals.to(tl.float64), u_vals.to(tl.float64), allow_tf32=False).to(
        LU_ptr.dtype.element_ty
    )
    tl.store(LU_ptr + tile_offs, tile - update, mask=tile_mask)


def _lu_factor_parallel(A):
    """Parallel physical-layout blocked LU.  Returns (LU, pivots, info) with
    0-based IPIV pivots.  ``LU`` is fp64-accurate for fp64 input (~1e-14
    backward error), so the inverse built from it needs no Newton refinement."""
    R = _LU_PAR_R
    A = A.contiguous()
    M = A.shape[-1]
    N = A.shape[-1]
    batch = A.numel() // (M * N)
    LU = A.reshape(batch, M, N).clone()
    pivots = torch.empty(batch, M, dtype=torch.int32, device=A.device)
    info = torch.zeros(batch, dtype=torch.int32, device=A.device)
    sval = torch.zeros(batch * (2 * R + 8), dtype=torch.float32, device=A.device)
    srow = torch.zeros(batch * (2 * R + 8), dtype=torch.int32, device=A.device)
    spiv = torch.zeros(batch, dtype=A.dtype, device=A.device)
    barrier = torch.zeros(batch * 8, dtype=torch.int32, device=A.device)

    for k0 in range(0, N, _LU_PAR_PANEL):
        p = min(_LU_PAR_PANEL, N - k0)
        group = triton.next_power_of_2(triton.cdiv(M - k0, R))
        col_block = triton.next_power_of_2(p)
        _lu_panel_par[(R, batch)](
            LU,
            pivots,
            info,
            sval,
            srow,
            spiv,
            barrier,
            k0,
            M,
            N,
            p,
            R,
            group,
            col_block,
            num_warps=4,
        )
        if k0 > 0:
            _lu_apply_left_par[(triton.cdiv(k0, _LU_PAR_TILE_N), batch)](
                LU,
                pivots,
                k0,
                M,
                N,
                p,
                _LU_PAR_TILE_N,
                num_warps=4,
            )
        trailing_n = N - k0 - p
        if trailing_n > 0:
            _lu_swap_right_solve_par[(triton.cdiv(trailing_n, _LU_PAR_TILE_N), batch)](
                LU,
                pivots,
                k0,
                M,
                N,
                p,
                col_block,
                _LU_PAR_TILE_N,
                num_warps=4,
            )
            _lu_trailing_update_par[
                (
                    triton.cdiv(trailing_n, _LU_PAR_TILE_M),
                    triton.cdiv(trailing_n, _LU_PAR_TILE_N),
                    batch,
                )
            ](
                LU,
                k0,
                M,
                N,
                p,
                _LU_PAR_TILE_M,
                _LU_PAR_TILE_N,
                p,
                num_warps=4,
            )
    return LU.reshape(A.shape), pivots, info


# ---------------------------------------------------------------------------
# thead (PPU) large-M variants: the grid-barrier parallel LU above deadlocks
# and wide fp64 tl.dots are nondeterministic on thead (policy table), so its
# M > 64 factorization / solves / powers use barrier-free kernels whose every
# fp64 dot is capped at a (16,16,16) tile.  These run only on thead.
# ---------------------------------------------------------------------------


@triton.jit
def _thead_lu_update16_par(
    LU_ptr,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
):
    """A[K0+PANEL:M, K0+PANEL:N] -= L @ U for the thead large-M LU — the
    trailing update of _lu_trailing_update_par with fp64 tl.dots capped at
    (32,32,32) tiles (thead's fp64 dots with wider tiles / more warps
    sporadically return garbage once magnitudes grow; 4-warp dots up to
    (64,64,64) were bitwise-deterministic in the envelope probe).  Grid:
    (row-tiles, col-tiles, batch)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    rows = K0 + PANEL + pid_m * 32 + tl.arange(0, 32)
    cols = K0 + PANEL + pid_n * 32 + tl.arange(0, 32)
    kr32 = tl.arange(0, 32)
    base = pid_b * M * N
    tile_offs = base + rows[:, None] * N + cols[None, :]
    tile_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tile = tl.load(LU_ptr + tile_offs, mask=tile_mask, other=0.0).to(
        LU_ptr.dtype.element_ty
    )
    acc = tl.zeros((32, 32), dtype=tl.float64)
    for kb in range(0, PANEL, 32):
        kr = K0 + kb + kr32
        k_mask = kr < K0 + PANEL
        l32 = tl.load(
            LU_ptr + base + rows[:, None] * N + kr[None, :],
            mask=(rows[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        u32 = tl.load(
            LU_ptr + base + kr[:, None] * N + cols[None, :],
            mask=k_mask[:, None] & (cols[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(l32.to(tl.float64), u32.to(tl.float64), allow_tf32=False)
    tl.store(
        LU_ptr + tile_offs,
        (tile - acc.to(LU_ptr.dtype.element_ty)),
        mask=tile_mask,
    )


@triton.jit
def _thead_lu_panel_serial(
    LU_ptr,
    pivots_ptr,
    info_ptr,
    K0: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    """Barrier-free panel LU factorization for thead (one program per matrix;
    the R-group barrier kernel _lu_panel_par deadlocks there).  Serial
    right-looking elimination over the PANEL columns of [K0, M) x [K0, K0+PANEL):
    per column — pivot search over the column vector in registers, physical row
    swap over the panel columns, scale the below-diagonal entries, then the
    rank-1 update over ROW_TILE-row chunks (register-bounded).  All scalars /
    reductions stay inside one CTA, so no grid barrier is needed and the result
    is deterministic.  Stores 0-based IPIV pivots."""
    pid = tl.program_id(0)
    base = pid * M * N
    allr = tl.arange(0, BLOCK_M)
    allc = tl.arange(0, COL_BLOCK)
    rt = tl.arange(0, ROW_TILE)
    info_val = 0
    for jj in tl.range(0, PANEL):
        j = K0 + jj
        # 1. current column j (rows >= j) — pivot search in registers.
        col = tl.load(
            LU_ptr + base + (j + allr) * N + j, mask=(j + allr) < M, other=0.0
        )
        ac = tl.where((j + allr) < M, tl.abs(col), -1.0)
        pmax = tl.max(ac, axis=0)
        p = j + tl.min(tl.where(ac == pmax, allr, BLOCK_M), axis=0)
        pivot = tl.load(LU_ptr + base + p * N + j)
        if pmax != pmax or pmax == 0.0:
            if info_val == 0:
                info_val = j + 1
        tl.store(pivots_ptr + pid * M + j, p)
        # 2. swap rows j and p over the panel columns.
        if p != j:
            rj = tl.load(
                LU_ptr + base + j * N + K0 + allc, mask=allc < PANEL, other=0.0
            )
            rp = tl.load(
                LU_ptr + base + p * N + K0 + allc, mask=allc < PANEL, other=0.0
            )
            tl.store(LU_ptr + base + j * N + K0 + allc, rp, mask=allc < PANEL)
            tl.store(LU_ptr + base + p * N + K0 + allc, rj, mask=allc < PANEL)
        # Cross-thread dependency through global memory inside this CTA: rows
        # written by one thread are read by others in the next step, so each
        # store phase must be fenced before the dependent loads (same pattern
        # as the chained in-place swaps in _lu_apply_left_par /
        # _lu_swap_right_solve_par — without the barrier Triton may reorder
        # the loads ahead of the stores and the factorization races).
        tl.debug_barrier()
        # 3. scale the below-diagonal entries of column j.
        s = tl.load(
            LU_ptr + base + (j + 1 + allr) * N + j,
            mask=(j + 1 + allr) < M,
            other=0.0,
        )
        s = s / pivot
        tl.store(
            LU_ptr + base + (j + 1 + allr) * N + j,
            s,
            mask=(j + 1 + allr) < M,
        )
        tl.debug_barrier()
        # 4. rank-1 update rows (j, M) x cols (jj+1, PANEL), in row chunks.
        if jj + 1 < PANEL:
            ux = tl.load(
                LU_ptr + base + j * N + (K0 + jj + 1 + allc),
                mask=(jj + 1 + allc) < PANEL,
                other=0.0,
            )
            for r0 in tl.range(j + 1, M, ROW_TILE):
                rows_r = r0 + rt
                rmask = rows_r < M
                sc = tl.load(LU_ptr + base + rows_r * N + j, mask=rmask, other=0.0)
                xmask = (jj + 1 + allc) < PANEL
                blk = tl.load(
                    LU_ptr + base + rows_r[:, None] * N + (K0 + jj + 1 + allc)[None, :],
                    mask=rmask[:, None] & xmask[None, :],
                    other=0.0,
                )
                blk = blk - sc[:, None] * ux[None, :]
                tl.store(
                    LU_ptr + base + rows_r[:, None] * N + (K0 + jj + 1 + allc)[None, :],
                    blk,
                    mask=rmask[:, None] & xmask[None, :],
                )
        # fence this column's stores before the next column's pivot search
        tl.debug_barrier()
    tl.store(info_ptr + pid, info_val)


def _thead_lu_factor(A):
    """Blocked LU for thead (M > 64), fully barrier-free and deterministic:
    the panel factorization is a single CTA per matrix (_thead_lu_panel_serial —
    the R-group barrier kernel deadlocks on thead), the row swaps / solve and
    the trailing update are launch-ordered kernels whose fp64 tl.dots are
    capped at (32,32,32) 4-warp tiles (the shape envelope thead's fp64 dot
    stays deterministic in once magnitudes grow — see the probe evidence in
    the _thead_fp64_mm16_kernel note).  Returns (LU, pivots, info) with
    0-based IPIV pivots."""
    A = A.contiguous()
    M = A.shape[-1]
    N = A.shape[-1]
    batch = A.numel() // (M * N)
    LU = A.reshape(batch, M, N).clone()
    pivots = torch.empty(batch, M, dtype=torch.int32, device=A.device)
    info = torch.zeros(batch, dtype=torch.int32, device=A.device)

    for k0 in range(0, N, _LU_PAR_PANEL):
        p = min(_LU_PAR_PANEL, N - k0)
        col_block = triton.next_power_of_2(p)
        _thead_lu_panel_serial[(batch,)](
            LU,
            pivots,
            info,
            k0,
            M,
            N,
            p,
            triton.next_power_of_2(M),
            col_block,
            _LU_PAR_TILE_M // 2,
            num_warps=8,
        )
        if k0 > 0:
            _lu_apply_left_par[(triton.cdiv(k0, _LU_PAR_TILE_N), batch)](
                LU,
                pivots,
                k0,
                M,
                N,
                p,
                _LU_PAR_TILE_N,
                num_warps=4,
            )
        trailing_n = N - k0 - p
        if trailing_n > 0:
            _lu_swap_right_solve_par[(triton.cdiv(trailing_n, _LU_PAR_TILE_N), batch)](
                LU,
                pivots,
                k0,
                M,
                N,
                p,
                col_block,
                _LU_PAR_TILE_N,
                num_warps=4,
            )
            _thead_lu_update16_par[
                (
                    triton.cdiv(trailing_n, 32),
                    triton.cdiv(trailing_n, 32),
                    batch,
                )
            ](LU, k0, M, N, p, num_warps=4)
    return LU.reshape(A.shape), pivots, info


def _thead_inverse_large(A):
    """A^-1 for M > 64 on thead, deterministic: parallel LU whose only fp64
    tl.dots are (16,16,16) tiles (_thead_lu_factor), then the scalar
    forward/backward substitution solve (no tl.dot at all).  fp64 storage
    throughout — no fp64 dot ever exceeds the tile shape that is reliable on
    this platform."""
    m = A.shape[-1]
    A3 = A.reshape(-1, m, m)
    batch = A3.shape[0]
    LU, pivots, info = _thead_lu_factor(A3)
    if torch.any(info != 0):
        raise RuntimeError(
            "linalg_matrix_power: the input matrix is singular (LU factorization "
            "encountered a zero pivot)"
        )
    # 0-based IPIV -> 1-based -> row-permutation index, then solve with the
    # row-permuted identity: A = P L U, so L U X = P solves A X = I.
    perm = _pivots_to_perm_gpu(pivots + 1, m)  # (batch, m), int64
    eye = torch.eye(m, dtype=A.dtype, device=A.device)
    if batch == 1:
        pb = eye[perm[0]].reshape(1, m, m)
    else:
        pb = torch.gather(
            eye.expand(batch, m, m).contiguous(),
            1,
            perm.unsqueeze(-1).expand(batch, m, m),
        )
    # Blocked triangular solve in place, mirroring the shared lu_solve flow
    # (unit-lower L forward, then upper U backward) with _trsm_kernel's D16
    # mode — every fp64 tl.dot stays a (16,16,16) tile, the only fp64-dot
    # shape that is deterministic on thead once magnitudes grow.  (The legacy
    # scalar substitution kernels ran one serial O(M^2) loop per RHS column,
    # ~50x slower at M=1024.)
    X = pb.clone()
    for b in range(batch):
        _trsm_solve_2d(LU[b], X[b], upper=False, unitriangular=True, d16=True)
        _trsm_solve_2d(LU[b], X[b], upper=True, unitriangular=False, d16=True)
    return X.reshape(A.shape)


def _thead_fp64_power_large(X, k, shape):
    """X^k for M > 64 on thead, deterministic: host binary exponentiation
    whose matmuls are the (16,16,16)-tile _thead_fp64_mm16."""
    m = X.shape[-1]
    X2 = X.reshape(-1, m, m)
    res = None
    while k > 0:
        if k & 1:
            res = X2 if res is None else _thead_fp64_mm16(res, X2)
        k >>= 1
        if k > 0:
            X2 = _thead_fp64_mm16(X2, X2)
    return res.reshape(shape)


def _inverse(A: torch.Tensor, use_df64=False, A_l=None) -> torch.Tensor:
    """A⁻¹ entirely on flag_gems Triton kernels: LU factorization + solve.

    flag_gems has no linalg_inv op, and torch.linalg.inv under use_gems would
    dispatch into the registered LU stack (linalg_lu_factor_ex_out), so the
    factorization uses the flag_gems kernel directly and the solve runs the
    tiled TRSM op in ``linalg_lu_solve`` below.  ``use_df64`` selects the df64
    (double-single) kernels for no-fp64 backends.
    """
    LU, LU_L, pivots, info, perm = _lu_factor_ex_local(A, use_df64, A_l)
    if info is not None and torch.any(info != 0):
        raise RuntimeError(
            "linalg_matrix_power: the input matrix is singular (LU factorization "
            "encountered a zero pivot)"
        )
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    if A.ndim > 2:
        eye = eye.expand(A.shape[:-2] + (A.shape[-1], A.shape[-1])).contiguous()
    res = linalg_lu_solve(
        LU, perm, eye, left=True, adjoint=False, LU_L=LU_L, use_df64=use_df64
    )
    if use_df64:
        return res  # (X, X_l) df64 pair
    # f32 inputs compute the inverse in f32 (fp32 storage + fp64 accumulation in
    # the LU/TRSM updates); the fp32-stored LU factors limit the TRSM solve to
    # ~1e-7 accuracy, so one fp64-accumulation Newton step brings the inverse to
    # the fp32-representable floor (~6e-8) — needed to meet the f32 tolerances
    # once the result is raised to the |n|th power.  fp64 inputs need no
    # refinement (the parallel LU is fp64-accurate).
    if A.dtype == torch.float32:
        # One step suffices for small M (the local fp64-accumulating LU is more
        # accurate); the fp32-stored parallel LU for large M has ~1e-6 backward
        # error, so a second step is needed to avoid a marginal element once the
        # inverse is raised to |n|.
        iters = 1 if A.shape[-1] <= 512 else 2
        res = _newton_refine(A, res, iters=iters)
    return res


def _eye_like(A: torch.Tensor) -> torch.Tensor:
    m = A.shape[-1]
    shape = A.shape
    eye = torch.eye(m, dtype=A.dtype, device=A.device)
    if len(shape) > 2:
        eye = eye.expand(shape[:-2] + (m, m)).clone()
    return eye


# ===========================================================================
# Main entry point
# ===========================================================================


def linalg_matrix_power(
    A: torch.Tensor,
    n: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    logger.debug("GEMS LINALG_MATRIX_POWER")

    # ---- validation ----
    shape = A.shape
    if len(shape) < 2:
        raise RuntimeError(
            f"linalg_matrix_power: A must be at least 2-D, got shape {shape}"
        )
    m, k = shape[-2], shape[-1]
    if m != k:
        raise RuntimeError(f"linalg_matrix_power: A must be square, got ({m}, {k})")
    if not isinstance(n, int):
        raise TypeError(f"linalg_matrix_power: n must be int, got {type(n).__name__}")
    if A.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only float32 and float64, "
            f"got {A.dtype}"
        )

    # ---- n == 0 ----
    if n == 0:
        eye = _eye_like(A)
        if out is not None:
            out.copy_(eye)
            return out
        return eye

    # ---- n == 1: A¹ = A — plain copy, no computation / kernel launch ----
    if n == 1:
        if out is not None:
            out.copy_(A)
            return out
        return A.clone()

    # ---- flag_gems Triton kernels are CUDA-only; n==0/n==1 above work on any
    # device, every computational path below requires the flag_gems device. ----
    if A.device.type != flag_gems.device:
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only {flag_gems.device}, "
            f"got {A.device}"
        )

    # ---- negative n ----
    upcast = False
    # On backends without fp64 (e.g. ascend, iluvatar) the fp64 path is
    # unavailable, so f32 negatives use the df64 (double-single) kernels:
    # df64 inverse (hi/lo) + df64 binary-exponentiation power.  thead (PPU) is
    # fp64-capable but its fp64 tl.dot nondeterministically returns garbage
    # once operand magnitudes grow (see the thead policy row), so its negative
    # powers run in df64 too — fp64 inputs are split into an fp32 (hi, lo)
    # pair first and the df64 result is recombined into fp64.
    use_df64 = A.dtype == torch.float32 and (
        not flag_gems.runtime.device.support_fp64 or _PLATFORM.thead
    )
    if n < 0 and _PLATFORM.thead:
        # thead negative powers are fully deterministic by construction:
        # pure-fp32 df64 arithmetic for M <= 64 (in-register kernels), and an
        # fp64 chain whose every tl.dot is a (16,16,16) tile for larger M —
        # the platform's fp64 dot sporadically returns garbage with wider
        # tiles once operand magnitudes grow (see the thead policy row).
        if m > _DF64_MAX_M:
            # M > 64: the in-register df64 kernels stop at 64, so run the
            # whole chain in fp64 through _thead_inverse_large /
            # _thead_fp64_mm16 (16-tile dots only).  fp32 inputs ride the same
            # fp64 chain and the result is cast back — the cast reproduces the
            # fp32-cast reference exactly, including +/-inf past fp32 range.
            A64 = A if A.dtype == torch.float64 else A.double()
            X = _thead_inverse_large(A64)
            hi = _thead_fp64_power_large(X, -n, shape)
            if A.dtype == torch.float32:
                hi = hi.float()
            if out is not None:
                out.copy_(hi)
                return out
            return hi
        if A.dtype == torch.float64:
            # Split into an fp32 (hi, lo) pair on the CPU — IEEE fp64 keeps the
            # split error-free — and invert the *pair* in df64: inverting the
            # hi part alone leaves the inverse wrong by ~cond(A)*2^-24 (~1e-6
            # on the cond-80 tests, over the fp64 rtol).
            A_cpu = A.detach().cpu().double()
            Ah = A_cpu.float()
            Al = (A_cpu - Ah.double()).float()
            Ah = Ah.to(device=A.device)
            Al = Al.to(device=A.device)
        else:
            # fp32 inputs are exact as the df64 pair's hi part (low part 0).
            Ah = A.contiguous()
            Al = None
        Xh, Xl = _inverse(Ah, use_df64=True, A_l=Al)
        k = -n
        # A df64 pair overflows fp32 once (A^-1)^k leaves ~3.4e38 (the
        # cond-80 inputs do that past |n| ~ 19).  Pre-scale the power's input
        # pair by an exact power of two so every binary-exponentiation
        # intermediate stays in fp32 range — possible for any k whose result
        # fp64 itself can represent — then recombine in fp64 and scale back
        # (both steps exact: powers of two).  fp32 results are cast from the
        # fp64 recombine, which also keeps the correct +/-inf sign past fp32
        # range (the fp32-cast reference is +/-inf there too).
        s = _df64_power_scale(Xh, k)
        hi, lo = _matrix_power_df64_pair(Xh, Xl, k, m, shape, scale=2.0**-s)
        res = hi.double() + lo.double()  # deterministic fp64 recombine
        if s:
            res = res * torch.pow(torch.tensor(2.0, dtype=torch.float64), k * s)
        if A.dtype == torch.float32:
            res = res.float()
        if out is not None:
            out.copy_(res)
            return out
        return res
    if n < 0 and use_df64:
        inv = _inverse(A, use_df64=True)
        if isinstance(inv, tuple):
            # Small M: df64 inverse (hi/lo) pair + df64 binary-exponentiation
            # power (the only supported df64 path — in-register kernels).
            Xh, Xl = inv
            return _matrix_power_df64(Xh, Xl, -n, m, shape, out=out)
        # Large M: the external fp32 LU (above the in-register df64 kernels)
        # has no df64 low part, so the inverse is a plain fp32 tensor.  Compute
        # the power in fp32 with the general dispatch below.
        A = inv
        n = -n
    elif n < 0:
        # f32 negatives: matrices above 512 use the f32-compute path — the
        # inverse is computed in f32 (fp32 storage + fp64 accumulation in the
        # LU/TRSM updates, fp64-accumulation Newton refinement), which is faster
        # once the barrier-bound LU is amortised.  Smaller f32 matrices keep the
        # fp64 upcast (the refinement overhead is not amortised there).
        # The power (A⁻¹)^|n| always runs in fp64 and the result is cast to f32.
        upcast = A.dtype == torch.float32
        if upcast and A.shape[-1] <= 512:
            A = A.double()
        A = _inverse(A)
        n = -n
        if upcast:
            A = A.double()

    # ---- n == 2, 3: fast paths for large M ----
    if n == 2 and m > TRITON_THRESHOLD:
        r = _matmul(A, A)
        if upcast:
            r = r.float()
        if out is not None:
            out.copy_(r)
            return out
        return r
    if n == 3 and m > TRITON_THRESHOLD:
        r = _matmul(_matmul(A, A), A)
        if upcast:
            r = r.float()
        if out is not None:
            out.copy_(r)
            return out
        return r

    # ---- flatten batch ----
    if len(shape) > 2:
        A_flat = A.reshape(-1, m, m)
    else:
        A_flat = A.unsqueeze(0)
    batch_size = A_flat.shape[0]
    batch_stride = m * m

    if out is not None:
        if upcast:
            # fp64 compute buffer (the kernels produce fp64); cast to fp32 out
            # at the end.
            out_flat = torch.empty(
                batch_size, m, m, dtype=torch.float64, device=A.device
            )
        else:
            out_flat = out.reshape(-1, m, m)
    else:
        out_flat = torch.empty(batch_size, m, m, dtype=A.dtype, device=A.device)

    # ---- dispatch ----
    if m <= SINGLE_TILE_MAX and A.device.type == flag_gems.device:
        # Tier 1: single-program fused (M <= 32).  tl.dot in sweet spot.
        BLOCK = max(triton.next_power_of_2(m), 16)
        _single_tile_kernel[(batch_size,)](
            A_flat,
            out_flat,
            m,
            n,
            batch_stride,
            BLOCK=BLOCK,
            FIX_FP64=_PLATFORM.metax_fp64_dot_fix and A.dtype == torch.float64,
            G=BLOCK // 16,
        )

    elif m <= TILED_MAX and A.device.type == flag_gems.device:
        # Tier 2: single-tile (33 <= M <= 64).
        # Grid-sync barrier overhead (~5 us/barrier × 3) exceeds the
        # single-SM tl.dot(64,64) time for 4-tile grids.  CUDA graph
        # memcpy overhead (~5 us × 3 copies) also dominates for M≤64.
        BLOCK = max(triton.next_power_of_2(m), 16)
        _single_tile_kernel[(batch_size,)](
            A_flat,
            out_flat,
            m,
            n,
            batch_stride,
            BLOCK=BLOCK,
            FIX_FP64=_PLATFORM.metax_fp64_dot_fix and A.dtype == torch.float64,
            G=BLOCK // 16,
        )

    elif _PLATFORM.grid_sync_fused and A.device.type == flag_gems.device and m <= 256:
        # Tier 3: grid-level sync fused (65 <= M <= 256).
        # Disabled on thead (grid_sync_fused policy): its grid spin-barrier
        # deadlocks, so 65 <= M <= 256 falls through to the host path below.
        TILES = triton.cdiv(m, TILE)
        # Fresh buffers per call: the kernel's Step 0 fully overwrites every
        # scratch slot it reads, and the round-based barrier logic works from a
        # zero-initialized counter (0 is a multiple of n_total), so nothing
        # needs to persist across calls.  Allocating fresh also keeps concurrent
        # calls on different streams from racing on shared buffers, and avoids
        # an ever-growing barrier counter that could overflow int32.
        scratch = torch.empty(4 * batch_size, m, m, dtype=A.dtype, device=A.device)
        barrier = torch.zeros(batch_size * 64, dtype=torch.int32, device=A.device)
        _grid_sync_kernel[(batch_size, TILES, TILES)](
            A_flat,
            out_flat,
            scratch,
            barrier,
            m,
            n,
            batch_stride,
            TILE_BLOCK=TILE,
            TILES=TILES,
            FIX_FP64=_PLATFORM.metax_fp64_dot_fix and A.dtype == torch.float64,
            G=TILE // 16,
        )

    else:
        # M > 256 — and, on thead, 65 <= M <= 256 (its grid-sync Tier 3 is
        # disabled by the grid_sync_fused policy): host-side binary
        # exponentiation with the flag_gems Triton matmul kernels (mm for 2D,
        # bmm for batched), one launch per step.
        is_batched = batch_size > 1
        z = A_flat if is_batched else A_flat.squeeze(0)
        result = None
        n_remaining = n
        while n_remaining > 0:
            if n_remaining & 1:
                result = z if result is None else _matmul(result, z)
            n_remaining >>= 1
            if n_remaining > 0:
                z = _matmul(z, z)
        if is_batched:
            out_flat.copy_(result)
        else:
            out_flat.squeeze_(0).copy_(result)

    # ---- reshape back ----
    if upcast:
        out_flat = out_flat.float()
    if len(shape) > 2:
        out_flat = out_flat.reshape(shape)
    else:
        out_flat = out_flat.squeeze(0)

    if out is not None:
        if upcast:
            out.copy_(out_flat)
        return out
    return out_flat
