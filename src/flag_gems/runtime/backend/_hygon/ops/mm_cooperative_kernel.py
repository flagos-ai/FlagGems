"""Cooperative GEMM using logical TL/TLE roles and automatic shared charts."""

from flag_gems.utils import libentry

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.language.extra.hip import accumulator_fence
from triton.language.extra.hip import cta_barrier as cta_phase


@triton.jit
def grouped_program_coordinates(
    pid, count_m: tl.constexpr, count_n: tl.constexpr, GROUP_M: tl.constexpr
):
    # The caller launches count_m * count_n consecutive program IDs.
    tl.static_assert(count_m > 0 and count_n > 0 and GROUP_M > 0)
    # Program IDs in the supported launch grid are nonnegative.
    pid = pid.to(tl.uint32)
    group_width: tl.constexpr = GROUP_M * count_n
    if count_n == 1:
        # Keep rotated tail-row ordering; the single column is always zero.
        if count_m > GROUP_M and count_m % GROUP_M != 0:
            first_tail: tl.constexpr = (count_m // GROUP_M) * GROUP_M
            tail_rows: tl.constexpr = count_m % GROUP_M
            pm = tl.where(pid >= first_tail, first_tail + pid % tail_rows, pid)
            return pm.to(tl.int32), tl.cast(0, tl.int32)
        return pid.to(tl.int32), tl.cast(0, tl.int32)
    elif count_m <= GROUP_M:
        return (pid % count_m).to(tl.int32), (pid // count_m).to(tl.int32)
    elif count_m % GROUP_M == 0:
        return ((pid // group_width) * GROUP_M + pid % GROUP_M).to(tl.int32), (
            (pid % group_width) // GROUP_M
        ).to(tl.int32)
    else:
        full_groups: tl.constexpr = count_m // GROUP_M
        full_span: tl.constexpr = full_groups * group_width
        tail_m: tl.constexpr = count_m % GROUP_M
        in_tail = pid >= full_span
        full_m = (pid // group_width) * GROUP_M + pid % GROUP_M
        full_n = (pid % group_width) // GROUP_M
        # Preserve the original absolute modulo, including rotated tail rows.
        tail_m_index = full_groups * GROUP_M + pid % tail_m
        tail_n = (pid - full_span) // tail_m
        return tl.where(in_tail, tail_m_index, full_m).to(tl.int32), tl.where(
            in_tail, tail_n, full_n
        ).to(tl.int32)


@triton.jit
def _coordinates(M: tl.constexpr, N: tl.constexpr, GROUP_M: tl.constexpr):
    if GROUP_M == 0:
        return tl.program_id(0), tl.program_id(1)
    else:
        cm: tl.constexpr = (M + 255) // 256
        cn: tl.constexpr = (N + 255) // 256
        return grouped_program_coordinates(
            tl.program_id(0) + tl.program_id(1) * cm, cm, cn, GROUP_M
        )


@triton.jit
def _produce_tile(
    B,
    owner,
    n,
    k,
    t,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    column = t * BK + k
    mask = n[None, :] < N
    value = tle.load(
        B + n[None, :].to(tl.int64) * K + column[:, None], mask, other=0, is_async=True
    )
    owner.slot(t % 2).store(value)
    cta_phase()
    cta_phase()


@triton.jit
def _producer(
    B,
    owner,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pm, pn = _coordinates(M, N, GROUP_M)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    for pair in range(K // (2 * BK)):
        _produce_tile(B, owner, n, k, pair * 2, N, K, BK)
        _produce_tile(B, owner, n, k, pair * 2 + 1, N, K, BK)
    if (K // BK) % 2:
        _produce_tile(B, owner, n, k, K // BK - 1, N, K, BK)
    cta_phase()
    cta_phase()


@triton.jit
def _load_parts(
    A, row, K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, M: tl.constexpr, t
):
    pointer = tl.make_block_ptr(A, (M, K), (K, 1), (row, t * BK), (BM, BK), (1, 0))
    full = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    lo, hi = tl.split(full.reshape((BM, 2, BK // 2)).permute((0, 2, 1)))
    return lo, hi


@triton.jit
def _compute_parts(
    A,
    owner,
    acc,
    av0,
    av1,
    row,
    t,
    M: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    HIGH_PHASE: tl.constexpr,
):
    bv0 = owner.slot(t % 2).subslice(0, BK // 2, 0).load()
    if HIGH_PHASE:
        cta_phase()
    acc = accumulator_fence(acc, 1)
    acc = tl.dot(av0, bv0, acc, allow_tf32=False)
    acc = accumulator_fence(acc, 0)
    if not HIGH_PHASE:
        cta_phase()
    next0, next1 = _load_parts(A, row, K, BM, BK, M, t + 1)
    bv1 = owner.slot(t % 2).subslice(BK // 2, BK // 2, 0).load()
    if HIGH_PHASE:
        cta_phase()
    acc = accumulator_fence(acc, 1)
    acc = tl.dot(av1, bv1, acc, allow_tf32=False)
    acc = accumulator_fence(acc, 0)
    if not HIGH_PHASE:
        cta_phase()
    return acc, next0, next1


@triton.jit
def _consumer(
    A,
    C,
    owner,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ROW_START: tl.constexpr,
    HIGH_PHASE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pm, pn = _coordinates(M, N, GROUP_M)
    row = pm * (2 * BM) + ROW_START
    acc = tl.zeros((BM, BN), tl.float32)
    av0, av1 = _load_parts(A, row, K, BM, BK, M, 0)
    cta_phase()
    cta_phase()
    for pair in range(tl.cdiv(K, BK) // 2):
        acc, av0, av1 = _compute_parts(
            A, owner, acc, av0, av1, row, pair * 2, M, K, BM, BK, HIGH_PHASE
        )
        acc, av0, av1 = _compute_parts(
            A, owner, acc, av0, av1, row, pair * 2 + 1, M, K, BM, BK, HIGH_PHASE
        )
    if tl.cdiv(K, BK) % 2:
        acc, av0, av1 = _compute_parts(
            A, owner, acc, av0, av1, row, tl.cdiv(K, BK) - 1, M, K, BM, BK, HIGH_PHASE
        )
    output = tl.make_block_ptr(C, (M, N), (N, 1), (row, pn * BN), (BM, BN), (1, 0))
    tl.store(output, acc.to(C.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.jit
def _mm_cooperative_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr = 256,
    BN: tl.constexpr = 256,
    BK: tl.constexpr = 64,
    GROUP_M: tl.constexpr = 0,
):
    tl.static_assert(BM == 256 and BN == 256 and BK == 64)
    tl.static_assert(K >= 64 and K < 8192 and K % 64 == 0)
    tl.static_assert(M > 0 and N > 0 and M < 2147483648 and N < 2147483648)
    tl.static_assert(512 * N < 2147483648 and GROUP_M >= 0)
    # Describe logical storage; the backend jointly selects its physical chart
    # from the global producer and actual MMAC consumers.
    owner = tle.gpu.alloc([2, BK, BN], A.dtype.element_ty)
    tle.gpu.warp_specialize(
        [
            (_consumer, (A, C, owner, M, N, K, BM // 2, BN, BK, 0, False, GROUP_M)),
            (
                _consumer,
                (A, C, owner, M, N, K, BM // 2, BN, BK, BM // 2, True, GROUP_M),
            ),
            (
                _producer,
                (B, owner, M, N, K, BN, BK, GROUP_M),
            ),
        ],
        [4, 4],
        [128, 32],
    )
