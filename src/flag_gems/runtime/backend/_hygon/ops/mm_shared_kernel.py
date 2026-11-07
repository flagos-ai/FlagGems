# Standard async group contract: each operand issues one committed group.
# Prefix has six groups; retiring the first pair leaves four younger groups.
# Each loop iteration retires a pair and issues a pair, again leaving four.
# Tail retires pairs with four, two, then zero outstanding younger groups.
# These are API group counts, not native vmcnt immediates. The existing async
# wait analysis derives instruction counts; no count crosses a consumed tile.
"""Four-slot f16/bf16 matmul with ordinary shared layouts and async groups."""

from flag_gems.utils import libentry

from .mm_shared_layout import (
    column_layout,
    grouped_program_coordinates,
    shared_frame_layout,
)

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import mfma
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.language import range as pipeline_range

# Shared memory and dependencies use ordinary Gluon types. The HCU target
# copy derives the hardware resource from the actual shared chart and bounds.
from triton.experimental.gluon.language.amd.cdna3 import buffer_store
from triton.experimental.gluon.language.amd.cdna4.async_copy import (
    buffer_load_to_shared,
    commit_group,
    wait_group,
)
from triton.experimental.gluon.language.hcu import sched_barrier, matrix_load_to_shared


@g.jit
def tile_views(
    owner, slot, LA: gl.constexpr, LB: gl.constexpr, AS: gl.constexpr, BS: gl.constexpr
):
    frame = owner.index(slot & 3)
    # Subslice offsets must be multiples of the sliced power-of-two extent.
    # Put the larger operand first so both views satisfy that ordinary rule.
    a_size: gl.constexpr = AS[0] * AS[1]
    b_size: gl.constexpr = BS[0] * BS[1]
    a_offset: gl.constexpr = b_size if a_size <= b_size else 0
    b_offset: gl.constexpr = 0 if a_size <= b_size else a_size
    av = frame.slice(a_offset, a_size)._reinterpret(owner.dtype, AS, LA)
    bv = frame.slice(b_offset, b_size)._reinterpret(owner.dtype, BS, LB)
    return av, bv


@g.jit
def shared_read(source, layout: gl.constexpr, completion):
    return source.load(layout, token=completion)


@g.constexpr_function
def vector_source_legal(shared, axis):
    return all(
        shared.offset_bases[i][axis] == 1 << i and shared.offset_bases[i][1 - axis] == 0
        for i in range(3)
    ) and all(b[axis] % 8 == 0 for b in shared.offset_bases[3:])


@g.jit
def standard_tile_copy(
    base,
    dest,
    origin0,
    origin1,
    S0: gl.constexpr,
    S1: gl.constexpr,
    STRIDE0: gl.constexpr,
    STRIDE1: gl.constexpr,
    USE_MATRIX_COPY: gl.constexpr,
):
    if USE_MATRIX_COPY:
        matrix_load_to_shared(
            dest, base, (S0, S1), (STRIDE0, STRIDE1), (origin0, origin1)
        )
        commit_group()
    else:
        shared: gl.constexpr = dest.layout
        count: gl.constexpr = dest.shape[0] * dest.shape[1]
        source: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
        ordinal = gl.arange(0, count, layout=source)
        x = gl.full((count,), 0, gl.int32, source)
        y = gl.full((count,), 0, gl.int32, source)
        for bit in gl.static_range(3, len(shared.offset_bases)):
            digit = (ordinal >> bit) & 1
            x = x ^ (digit * shared.offset_bases[bit][0])
            y = y ^ (digit * shared.offset_bases[bit][1])
        x = x + origin0
        y = y + origin1
        if STRIDE1 == 1:
            y = y + (ordinal & 7)
        else:
            x = x + (ordinal & 7)
        offsets = (x * STRIDE0 + y * STRIDE1).to(gl.int32)
        # Each hardware lane's 8 elements are a proven contiguous global vector.
        if STRIDE1 == 1:
            gl.static_assert(vector_source_legal(shared, 1))
            gl.static_assert(S1 % 8 == 0)
        else:
            gl.static_assert(STRIDE0 == 1)
            gl.static_assert(vector_source_legal(shared, 0))
            gl.static_assert(S0 % 8 == 0)
        # AxisInfo divisibility describes the first element of contiguous groups.
        # Each 8-element group is exactly aligned_base + [0..7]; both attributes
        # below are proved together, not a claim that every lane element is /8.
        offsets = gl.max_contiguous(gl.multiple_of(offsets, 8), 8)
        mask = (x >= 0) & (x < S0) & (y >= 0) & (y < S1)
        mask = gl.max_constancy(mask, 8)
        flat = dest._reinterpret(
            dest.dtype, [count], gl.SwizzledSharedLayout(1, 1, 1, [0])
        )
        buffer_load_to_shared(flat, base, offsets, mask, other=0)
        commit_group()


@g.jit
def issue_pair(
    A,
    B,
    av,
    bv,
    row_m,
    col_n,
    k_origin,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    STRIDE_AM: gl.constexpr,
    STRIDE_BK: gl.constexpr,
    STRIDE_BN: gl.constexpr,
    USE_MATRIX_COPY: gl.constexpr,
):
    standard_tile_copy(A, av, row_m, k_origin, M, K, STRIDE_AM, 1, USE_MATRIX_COPY)
    standard_tile_copy(
        B, bv, k_origin, col_n, K, N, STRIDE_BK, STRIDE_BN, USE_MATRIX_COPY
    )


@g.jit
def read_b_column_chart(
    view, DB: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr, completion
):
    physical: gl.constexpr = column_layout(gl.to_linear_layout(DB, [BK, BN]))
    values = view.load(physical, token=completion)
    values = gl.reshape(values, [BK, BN // 32, 16, 2])
    values = gl.permute(values, [0, 1, 3, 2])
    return gl.convert_layout(gl.reshape(values, [BK, BN]), DB)


@libentry()
@g.jit
def _mm_shared_kernel(
    A,
    B,
    C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    STRIDE_AM: gl.constexpr,
    STRIDE_AK: gl.constexpr,
    STRIDE_BK: gl.constexpr,
    STRIDE_BN: gl.constexpr,
    STRIDE_CM: gl.constexpr,
    STRIDE_CN: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    GROUP_M: gl.constexpr = 8,
    TILES_PER_WARP: gl.constexpr = (2, 2),
    OUTPUT_LAYOUT: gl.constexpr = 0,
    A_SHARED_LAYOUT: gl.constexpr = None,
    B_SHARED_LAYOUT: gl.constexpr = None,
    USE_MATRIX_COPY: gl.constexpr = True,
):
    # 0: absolute MMA stores; 1: tile-base MMA stores; 2: column-chart stores.
    gl.static_assert(OUTPUT_LAYOUT == 0 or OUTPUT_LAYOUT == 1 or OUTPUT_LAYOUT == 2)
    # Four exact tail tiles and byte-offset buffer stores require these bounds.
    # This kernel has no target-private layout or source-ownership protocol.
    gl.static_assert(M > 0 and N > 0 and K >= 4 * BK)
    gl.static_assert(STRIDE_AM == K and STRIDE_AK == 1)
    if STRIDE_BK == 1:
        gl.static_assert(STRIDE_BN == K)
    else:
        gl.static_assert(N % 8 == 0)
        gl.static_assert(STRIDE_BK == N and STRIDE_BN == 1)
    gl.static_assert(STRIDE_CM == N and STRIDE_CN == 1)
    if not USE_MATRIX_COPY:
        gl.static_assert(2 * M * K < 2147483648 and 2 * K * N < 2147483648)
    gl.static_assert(2 * M * N < 2147483648)
    mma: gl.constexpr = AMDMFMALayout(
        3,
        [16, 16, 16],
        True,
        [WARPS_M, WARPS_N],
        tiles_per_warp=[TILES_PER_WARP[0], TILES_PER_WARP[1]],
        mmac_layout=1,
    )
    dot_k_width: gl.constexpr = 4 if BK < 32 else 8
    da: gl.constexpr = gl.DotOperandLayout(0, mma, dot_k_width)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, dot_k_width)
    gl.static_assert(BM * BK <= 4096 and BK * BN <= 4096)
    gl.static_assert(K % (4 * BK) == 0)
    gl.static_assert(BM % 8 == 0 and BN % 8 == 0 and BK % 8 == 0)
    la: gl.constexpr = A_SHARED_LAYOUT
    lb: gl.constexpr = B_SHARED_LAYOUT
    pid = gl.program_id(0)
    count_m: gl.constexpr = (M + BM - 1) // BM
    count_n: gl.constexpr = (N + BN - 1) // BN
    pid_m, pid_n = grouped_program_coordinates(pid, count_m, count_n, GROUP_M)
    row_m = pid_m * BM
    col_n = pid_n * BN
    # Operand sizes are powers of two: twice the larger size is the smallest
    # power-of-two frame containing both. Views refer to disjoint real storage,
    # including small tiles; no target-specific compact-owner encoding is used.
    frame_elements: gl.constexpr = 2 * max(BM * BK, BK * BN)
    owner = gl.allocate_shared_memory(
        A.dtype.element_ty,
        [4, frame_elements],
        shared_frame_layout(frame_elements),
    )
    a0, b0 = tile_views(owner, 0, la, lb, (BM, BK), (BK, BN))
    a1, b1 = tile_views(owner, 1, la, lb, (BM, BK), (BK, BN))
    a2, b2 = tile_views(owner, 2, la, lb, (BM, BK), (BK, BN))
    a3, b3 = tile_views(owner, 3, la, lb, (BM, BK), (BK, BN))

    # Prefill three tile pairs, acquire tile0, then fill the fourth slot.
    issue_pair(
        A,
        B,
        a0,
        b0,
        row_m,
        col_n,
        0,
        M,
        N,
        K,
        STRIDE_AM,
        STRIDE_BK,
        STRIDE_BN,
        USE_MATRIX_COPY,
    )
    issue_pair(
        A,
        B,
        a1,
        b1,
        row_m,
        col_n,
        BK,
        M,
        N,
        K,
        STRIDE_AM,
        STRIDE_BK,
        STRIDE_BN,
        USE_MATRIX_COPY,
    )
    issue_pair(
        A,
        B,
        a2,
        b2,
        row_m,
        col_n,
        2 * BK,
        M,
        N,
        K,
        STRIDE_AM,
        STRIDE_BK,
        STRIDE_BN,
        USE_MATRIX_COPY,
    )
    completion = wait_group(4, return_token=True)
    acurrent = shared_read(a0, da, completion)
    if OUTPUT_LAYOUT == 2:
        bcurrent = read_b_column_chart(b0, db, BK, BN, completion)
    else:
        bcurrent = shared_read(b0, db, completion)
    issue_pair(
        A,
        B,
        a3,
        b3,
        row_m,
        col_n,
        3 * BK,
        M,
        N,
        K,
        STRIDE_AM,
        STRIDE_BK,
        STRIDE_BN,
        USE_MATRIX_COPY,
    )
    accumulator = gl.full((BM, BN), 0, gl.float32, mma)
    last_slot = gl.cast(3, gl.int32)

    # Ordinary memdesc views and register tiles follow the four-slot ring.
    for start_k in pipeline_range(0, K - 4 * BK, BK, loop_unroll_factor=4):
        completion = wait_group(4, return_token=True)
        anext = shared_read(a1, da, completion)
        if OUTPUT_LAYOUT == 2:
            bnext = read_b_column_chart(b1, db, BK, BN, completion)
        else:
            bnext = shared_read(b1, db, completion)
        last_slot = (last_slot + 1) & 3
        anew, bnew = tile_views(owner, last_slot, la, lb, (BM, BK), (BK, BN))
        sched_barrier()
        issue_pair(
            A,
            B,
            anew,
            bnew,
            row_m,
            col_n,
            start_k + 4 * BK,
            M,
            N,
            K,
            STRIDE_AM,
            STRIDE_BK,
            STRIDE_BN,
            USE_MATRIX_COPY,
        )
        accumulator = mfma(acurrent, bcurrent, accumulator)
        a1, a2, a3 = a2, a3, anew
        b1, b2, b3 = b2, b3, bnew
        acurrent = anext
        bcurrent = bnext

    # Drain the four remaining tiles in order.
    completion = wait_group(4, return_token=True)
    anext = shared_read(a1, da, completion)
    if OUTPUT_LAYOUT == 2:
        bnext = read_b_column_chart(b1, db, BK, BN, completion)
    else:
        bnext = shared_read(b1, db, completion)
    accumulator = mfma(acurrent, bcurrent, accumulator)
    completion = wait_group(2, return_token=True)
    anext2 = shared_read(a2, da, completion)
    if OUTPUT_LAYOUT == 2:
        bnext2 = read_b_column_chart(b2, db, BK, BN, completion)
    else:
        bnext2 = shared_read(b2, db, completion)
    accumulator = mfma(anext, bnext, accumulator)
    completion = wait_group(0, return_token=True)
    anext3 = shared_read(a3, da, completion)
    if OUTPUT_LAYOUT == 2:
        bnext3 = read_b_column_chart(b3, db, BK, BN, completion)
    else:
        bnext3 = shared_read(b3, db, completion)
    accumulator = mfma(anext2, bnext2, accumulator)
    accumulator = mfma(anext3, bnext3, accumulator)
    owner._keep_alive()

    if OUTPUT_LAYOUT == 2:
        gl.static_assert(BN >= 32 and BN % 32 == 0)
        output_layout: gl.constexpr = column_layout(
            gl.to_linear_layout(mma, [BM, BN]), True
        )
        output = gl.reshape(accumulator, [BM, BN // 32, 2, 16])
        output = gl.permute(output, [0, 1, 3, 2])
        output = gl.convert_layout(gl.reshape(output, [BM, BN]), output_layout).to(
            C.dtype.element_ty
        )
        local_r = gl.arange(0, BM, layout=gl.SliceLayout(1, output_layout))
        local_c = gl.arange(0, BN, layout=gl.SliceLayout(0, output_layout))
        cr = row_m + local_r
        cn = col_n + local_c
        base = C + row_m * STRIDE_CM + col_n * STRIDE_CN
        offsets = (local_r[:, None] * STRIDE_CM + local_c[None, :] * STRIDE_CN).to(
            gl.int32
        )
        if M % BM == 0 and N % BN == 0:
            buffer_store(output, base, offsets)
        else:
            buffer_store(output, base, offsets, (cr[:, None] < M) & (cn[None, :] < N))
    elif OUTPUT_LAYOUT == 1:
        local_r = gl.arange(0, BM, layout=gl.SliceLayout(1, mma))
        local_c = gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
        cr = row_m + local_r
        cn = col_n + local_c
        output_base = C + row_m * STRIDE_CM + col_n * STRIDE_CN
        offsets = (local_r[:, None] * STRIDE_CM + local_c[None, :] * STRIDE_CN).to(
            gl.int32
        )
        mask = (cr[:, None] < M) & (cn[None, :] < N)
        buffer_store(accumulator.to(C.dtype.element_ty), output_base, offsets, mask)
    else:
        cr = row_m + gl.arange(0, BM, layout=gl.SliceLayout(1, mma))
        cn = col_n + gl.arange(0, BN, layout=gl.SliceLayout(0, mma))
        offsets = (cr[:, None] * STRIDE_CM + cn[None, :] * STRIDE_CN).to(gl.int32)
        mask = (cr[:, None] < M) & (cn[None, :] < N)
        buffer_store(accumulator.to(C.dtype.element_ty), C, offsets, mask)
