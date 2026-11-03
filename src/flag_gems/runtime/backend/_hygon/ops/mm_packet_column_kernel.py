"""Dense BF16 NN matmul with explicit bounded four-slot HCU packet transfers."""
from flag_gems.utils import libentry

from .mm_packet_mapping import grouped_program_coordinates
from .mm_packet_column_layout import column_layout, read_b_column_chart

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import mfma
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.language import range as pipeline_range
from triton.experimental.gluon.language.hcu import (
    WholePacketLayout as _WholePacketLayout,
    whole_packet_owner, packet_copy,
    packet_read, packet_commit, packet_wait, packet_sched_barrier,
    packet_buffer_store,
)


# Exact existing source chart: ordinal=((copy*warps+wave)*64+lane)*8+element.
# These map logical producer layout to hardware F. F itself is not GF(2).
A_SOURCE_BASIS = gl.constexpr((1, 2, 4, 8, 16, 32, 512, 1024, 2048, 64, 128, 256))
B_SOURCE_BASIS = gl.constexpr((1, 2, 4, 8, 16, 32, 64, 128, 512, 256, 1024, 2048))
WholePacketLayout = gl.constexpr(_WholePacketLayout)




@g.jit
def packet_views(owner, slot, LA: gl.constexpr, LB: gl.constexpr):
    frame = owner.index(slot & 3)
    av = frame.slice(4096, 4096)._reinterpret(gl.bfloat16, LA.tile_shape, LA)
    bv = frame.slice(0, 4096)._reinterpret(gl.bfloat16, LB.tile_shape, LB)
    return av, bv


@g.jit
def issue_pair(A, B, av, bv, row_m, col_n, k_origin,
               M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
               STRIDE_AM: gl.constexpr, STRIDE_BK: gl.constexpr):
    # Disabling a copy still writes a complete zero packet via NR=0. Never
    # skip a DMA and leave a prior ring slot's contents visible.
    enabled = (k_origin >= 0) & (k_origin < K)
    a = packet_copy(A, av, (M, K), (STRIDE_AM, 1),
                    (row_m, k_origin, enabled), A_SOURCE_BASIS)
    ac = packet_commit((a,))
    b = packet_copy(B, bv, (K, N), (STRIDE_BK, 1),
                    (k_origin, col_n, enabled), B_SOURCE_BASIS)
    bc = packet_commit((b,))
    return ac, bc


@libentry()
@g.jit
def _mm_packet_kernel(
    A, B, C,
    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    STRIDE_AM: gl.constexpr, STRIDE_AK: gl.constexpr,
    STRIDE_BK: gl.constexpr, STRIDE_BN: gl.constexpr,
    STRIDE_CM: gl.constexpr, STRIDE_CN: gl.constexpr,
    GROUP_M: gl.constexpr = 8,
    ENTRY_FAMILY: gl.constexpr = 0,
):
    # Existing whole-packet and StructuredDomain semantics, not a model-shape
    # dispatch. Unsupported inputs retain the ordinary host path.
    gl.static_assert(M > 0 and N > 0 and K >= 256 and K <= 4096)
    gl.static_assert((K & (K - 1)) == 0 and K % 16 == 0)
    gl.static_assert(N % 16 == 0)
    gl.static_assert(STRIDE_AM == K and STRIDE_AK == 1)
    gl.static_assert(STRIDE_BK == N and STRIDE_BN == 1)
    gl.static_assert(STRIDE_CM == N and STRIDE_CN == 1)
    gl.static_assert(2 * M * K < 2147483648 and 2 * K * N < 2147483648)
    gl.static_assert(2 * M * N < 2147483648)
    mma: gl.constexpr = AMDMFMALayout(3, [16, 16, 16], True, [4, 2],
                                     tiles_per_warp=[2, 2], mmac_layout=1)
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    la: gl.constexpr = WholePacketLayout(0, (256, 16))
    lb: gl.constexpr = WholePacketLayout(1, (16, 256))
    pid = gl.program_id(0)
    count_m: gl.constexpr = (M + 255) // 256
    count_n: gl.constexpr = (N + 255) // 256
    pid_m, pid_n = grouped_program_coordinates(pid, count_m, count_n, GROUP_M)
    row_m = pid_m * 256
    col_n = pid_n * 256
    owner = whole_packet_owner(gl.bfloat16, 4)
    a0, b0 = packet_views(owner, 0, la, lb)
    a1, b1 = packet_views(owner, 1, la, lb)
    a2, b2 = packet_views(owner, 2, la, lb)
    a3, b3 = packet_views(owner, 3, la, lb)

    # Prefix: three tile pairs, publish/read tile0, then issue tile3. This is
    # the original mode7 prefix, not a new stage count or pipeline schedule.
    ta0, tb0 = issue_pair(A, B, a0, b0, row_m, col_n, 0,
                          M, N, K, STRIDE_AM, STRIDE_BK)
    ta1, tb1 = issue_pair(A, B, a1, b1, row_m, col_n, 16,
                          M, N, K, STRIDE_AM, STRIDE_BK)
    ta2, tb2 = issue_pair(A, B, a2, b2, row_m, col_n, 32,
                          M, N, K, STRIDE_AM, STRIDE_BK)
    packet_wait((ta0, tb0), 4)
    acurrent = packet_read(a0, da, A_SOURCE_BASIS)
    bcurrent = read_b_column_chart(b0, db, B_SOURCE_BASIS, 16, 256)
    ta3, tb3 = issue_pair(A, B, a3, b3, row_m, col_n, 48,
                          M, N, K, STRIDE_AM, STRIDE_BK)
    accumulator = gl.full((256, 256), 0, gl.float32, mma)
    last_slot = gl.cast(3, gl.int32)

    # Queue carries are ordinary Gluon values: six async tokens, six memdesc
    # views, current A/B register tiles, accumulator, and bounded ring slot.
    # No extra CTAs or partial-LGKM claims are introduced by this source.
    for start_k in pipeline_range(0, K - 64, 16, loop_unroll_factor=4):
        packet_wait((ta1, tb1), 4)
        anext = packet_read(a1, da, A_SOURCE_BASIS)
        bnext = read_b_column_chart(b1, db, B_SOURCE_BASIS, 16, 256)
        last_slot = (last_slot + 1) & 3
        anew, bnew = packet_views(owner, last_slot, la, lb)
        packet_sched_barrier()
        tanew, tbnew = issue_pair(A, B, anew, bnew, row_m, col_n, start_k + 64,
                                  M, N, K, STRIDE_AM, STRIDE_BK)
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
    bnext = read_b_column_chart(b1, db, B_SOURCE_BASIS, 16, 256)
    accumulator = mfma(acurrent, bcurrent, accumulator)
    packet_wait((ta2, tb2), 0)
    anext2 = packet_read(a2, da, A_SOURCE_BASIS)
    bnext2 = read_b_column_chart(b2, db, B_SOURCE_BASIS, 16, 256)
    accumulator = mfma(anext, bnext, accumulator)
    packet_wait((ta3, tb3), 0)
    anext3 = packet_read(a3, da, A_SOURCE_BASIS)
    bnext3 = read_b_column_chart(b3, db, B_SOURCE_BASIS, 16, 256)
    accumulator = mfma(anext2, bnext2, accumulator)
    accumulator = mfma(anext3, bnext3, accumulator)
    owner._keep_alive()

    output_layout: gl.constexpr = column_layout(gl.to_linear_layout(mma, [256, 256]), True)
    output = gl.reshape(accumulator, [256,8,2,16])
    output = gl.permute(output, [0,1,3,2])
    output = gl.reshape(output, [256,256])
    output = gl.convert_layout(output, output_layout).to(gl.bfloat16)
    local_r = gl.arange(0, 256, layout=gl.SliceLayout(1, output_layout))
    local_c = gl.arange(0, 256, layout=gl.SliceLayout(0, output_layout))
    cr = row_m + local_r
    cn = col_n + local_c
    output_base = C + row_m * STRIDE_CM + col_n * STRIDE_CN
    offsets = (local_r[:, None] * STRIDE_CM + local_c[None, :] * STRIDE_CN).to(gl.int32)
    # Correct generated ceil-grid + unchanged grouped_program_coordinates
    # means every tile is full when both dimensions divide the tile extents.
    # For all other domains preserve the original absolute-coordinate mask.
    if M % 256 == 0 and N % 256 == 0:
        packet_buffer_store(output, output_base, offsets)
    else:
        mask = (cr[:, None] < M) & (cn[None, :] < N)
        packet_buffer_store(output, output_base, offsets, mask)
