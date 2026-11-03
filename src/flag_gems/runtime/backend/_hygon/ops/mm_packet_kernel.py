"""Four-slot BF16 packet matmul for contiguous NN or transposed-B storage."""
from flag_gems.utils import libentry

from .mm_packet_mapping import grouped_program_coordinates

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import mfma
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.language import range as pipeline_range
from triton.experimental.gluon.language.hcu import (
    WholePacketLayout as _WholePacketLayout, PacketPrefixLayout as _PacketPrefixLayout,
    PacketStorageLayout as _PacketStorageLayout,
    whole_packet_owner, packet_copy,
    packet_read, packet_commit, packet_wait, packet_sched_barrier,
    packet_buffer_store,
)


# Exact existing source chart: ordinal=((copy*warps+wave)*64+lane)*8+element.
# These map logical producer layout to hardware F. F itself is not GF(2).
WholePacketLayout = gl.constexpr(_WholePacketLayout)
PacketPrefixLayout = gl.constexpr(_PacketPrefixLayout)
PacketStorageLayout = gl.constexpr(_PacketStorageLayout)


@g.jit
def packet_views(owner, slot, LA: gl.constexpr, LB: gl.constexpr):
    frame = owner.index(slot & 3)
    av = frame.slice(4096, LA.tile_shape[0] * LA.tile_shape[1])._reinterpret(gl.bfloat16, LA.tile_shape, LA)
    bv = frame.slice(0, LB.tile_shape[0] * LB.tile_shape[1])._reinterpret(gl.bfloat16, LB.tile_shape, LB)
    return av, bv


@g.jit
def issue_pair(A, B, av, bv, row_m, col_n, k_origin,
               M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
               STRIDE_AM: gl.constexpr, STRIDE_BK: gl.constexpr, STRIDE_BN: gl.constexpr,
               A_SOURCE_BASIS: gl.constexpr, B_SOURCE_BASIS: gl.constexpr):
    # Disabling a copy still writes a complete zero packet via NR=0. Never
    # skip a DMA and leave a prior ring slot's contents visible.
    enabled = (k_origin >= 0) & (k_origin < K)
    a = packet_copy(A, av, (M, K), (STRIDE_AM, 1),
                    (row_m, k_origin, enabled), A_SOURCE_BASIS)
    ac = packet_commit((a,))
    if STRIDE_BK == 1:
        b = packet_copy(B, bv, (N, K), (STRIDE_BN, 1),
                        (col_n, k_origin, enabled), B_SOURCE_BASIS)
    else:
        b = packet_copy(B, bv, (K, N), (STRIDE_BK, 1),
                        (k_origin, col_n, enabled), B_SOURCE_BASIS)
    bc = packet_commit((b,))
    return ac, bc


@libentry()
@g.jit
def _mm_packet_rect_kernel(
    A, B, C,
    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    STRIDE_AM: gl.constexpr, STRIDE_AK: gl.constexpr,
    STRIDE_BK: gl.constexpr, STRIDE_BN: gl.constexpr,
    STRIDE_CM: gl.constexpr, STRIDE_CN: gl.constexpr,
    BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
    WARPS_M: gl.constexpr, WARPS_N: gl.constexpr,
    A_SOURCE_BASIS: gl.constexpr, B_SOURCE_BASIS: gl.constexpr,
    GROUP_M: gl.constexpr = 8,
    TILES_PER_WARP: gl.constexpr = (2, 2),
    WAIT_MODE: gl.constexpr = 0,
):
    # Existing whole-packet and StructuredDomain semantics, not a model-shape
    # dispatch. Unsupported inputs retain the ordinary host path.
    gl.static_assert(M > 0 and N > 0 and K >= 256 and K <= 4096)
    gl.static_assert((K & (K - 1)) == 0 and K % 16 == 0)
    gl.static_assert(STRIDE_AM == K and STRIDE_AK == 1)
    if STRIDE_BK == 1:
        gl.static_assert(STRIDE_BN == K)
    else:
        gl.static_assert(N % 16 == 0)
        gl.static_assert(STRIDE_BK == N and STRIDE_BN == 1)
    gl.static_assert(STRIDE_CM == N and STRIDE_CN == 1)
    gl.static_assert(2 * M * K < 2147483648 and 2 * K * N < 2147483648)
    gl.static_assert(2 * M * N < 2147483648)
    mma: gl.constexpr = AMDMFMALayout(3, [16, 16, 16], True, [WARPS_M, WARPS_N],
                                     tiles_per_warp=[TILES_PER_WARP[0], TILES_PER_WARP[1]], mmac_layout=1)
    dot_k_width: gl.constexpr = 4 if BK < 32 else 8
    da: gl.constexpr = gl.DotOperandLayout(0, mma, dot_k_width)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, dot_k_width)
    gl.static_assert(BM * BK <= 4096 and BK * BN <= 4096)
    gl.static_assert(K % (4 * BK) == 0)
    if STRIDE_BK == 1:
        la: gl.constexpr = PacketStorageLayout(0, (BM, BK), 1, (0, 1))
        lb: gl.constexpr = PacketStorageLayout(1, (BK, BN), 1, (1, 0))
    else:
        if BM * BK == 4096:
            la: gl.constexpr = WholePacketLayout(0, (BM, BK))
        else:
            la: gl.constexpr = PacketPrefixLayout(0, (BM, BK))
        if BK * BN == 4096:
            lb: gl.constexpr = WholePacketLayout(1, (BK, BN))
        else:
            lb: gl.constexpr = PacketPrefixLayout(1, (BK, BN))
    pid = gl.program_id(0)
    count_m: gl.constexpr = (M + BM - 1) // BM
    count_n: gl.constexpr = (N + BN - 1) // BN
    pid_m, pid_n = grouped_program_coordinates(pid, count_m, count_n, GROUP_M)
    row_m = pid_m * BM
    col_n = pid_n * BN
    owner = whole_packet_owner(gl.bfloat16, 4)
    a0, b0 = packet_views(owner, 0, la, lb)
    a1, b1 = packet_views(owner, 1, la, lb)
    a2, b2 = packet_views(owner, 2, la, lb)
    a3, b3 = packet_views(owner, 3, la, lb)

    # Prefix: three tile pairs, publish/read tile0, then issue tile3. This is
    # the original mode7 prefix, not a new stage count or pipeline schedule.
    ta0, tb0 = issue_pair(A, B, a0, b0, row_m, col_n, 0,
                          M, N, K, STRIDE_AM, STRIDE_BK, STRIDE_BN, A_SOURCE_BASIS, B_SOURCE_BASIS)
    ta1, tb1 = issue_pair(A, B, a1, b1, row_m, col_n, BK,
                          M, N, K, STRIDE_AM, STRIDE_BK, STRIDE_BN, A_SOURCE_BASIS, B_SOURCE_BASIS)
    ta2, tb2 = issue_pair(A, B, a2, b2, row_m, col_n, 2 * BK,
                          M, N, K, STRIDE_AM, STRIDE_BK, STRIDE_BN, A_SOURCE_BASIS, B_SOURCE_BASIS)
    packet_wait((ta0, tb0), 4)
    acurrent = packet_read(a0, da, A_SOURCE_BASIS)
    bcurrent = packet_read(b0, db, B_SOURCE_BASIS)
    ta3, tb3 = issue_pair(A, B, a3, b3, row_m, col_n, 3 * BK,
                          M, N, K, STRIDE_AM, STRIDE_BK, STRIDE_BN, A_SOURCE_BASIS, B_SOURCE_BASIS)
    accumulator = gl.full((BM, BN), 0, gl.float32, mma)
    last_slot = gl.cast(3, gl.int32)

    # Queue carries are ordinary Gluon values: six async tokens, six memdesc
    # views, current A/B register tiles, accumulator, and bounded ring slot.
    # No extra CTAs or partial-LGKM claims are introduced by this source.
    for start_k in pipeline_range(0, K - 4 * BK, BK, loop_unroll_factor=4):
        packet_wait((ta1, tb1), 4)
        anext = packet_read(a1, da, A_SOURCE_BASIS)
        bnext = packet_read(b1, db, B_SOURCE_BASIS)
        last_slot = (last_slot + 1) & 3
        anew, bnew = packet_views(owner, last_slot, la, lb)
        packet_sched_barrier()
        tanew, tbnew = issue_pair(A, B, anew, bnew, row_m, col_n, start_k + 4 * BK,
                                  M, N, K, STRIDE_AM, STRIDE_BK, STRIDE_BN, A_SOURCE_BASIS, B_SOURCE_BASIS)
        accumulator = mfma(acurrent, bcurrent, accumulator)
        ta1, ta2, ta3 = ta2, ta3, tanew
        tb1, tb2, tb3 = tb2, tb3, tbnew
        a1, a2, a3 = a2, a3, anew
        b1, b2, b3 = b2, b3, bnew
        acurrent = anext
        bcurrent = bnext

    # K>=256 gives the exact four remaining tiles. The original dynamic
    # counted-tail predicates are compile-time true, not runtime assumptions.
    packet_wait((ta1, tb1), 0)
    anext = packet_read(a1, da, A_SOURCE_BASIS)
    bnext = packet_read(b1, db, B_SOURCE_BASIS)
    accumulator = mfma(acurrent, bcurrent, accumulator)
    packet_wait((ta2, tb2), 0)
    anext2 = packet_read(a2, da, A_SOURCE_BASIS)
    bnext2 = packet_read(b2, db, B_SOURCE_BASIS)
    accumulator = mfma(anext, bnext, accumulator)
    packet_wait((ta3, tb3), 0)
    anext3 = packet_read(a3, da, A_SOURCE_BASIS)
    bnext3 = packet_read(b3, db, B_SOURCE_BASIS)
    accumulator = mfma(anext2, bnext2, accumulator)
    accumulator = mfma(anext3, bnext3, accumulator)
    owner._keep_alive()

    cr = row_m + gl.arange(0, BM, layout=gl.SliceLayout(1, mma))
    cn = col_n + gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
    offsets = (cr[:, None] * STRIDE_CM + cn[None, :] * STRIDE_CN).to(gl.int32)
    mask = (cr[:, None] < M) & (cn[None, :] < N)
    packet_buffer_store(accumulator.to(gl.bfloat16), C, offsets, mask)
