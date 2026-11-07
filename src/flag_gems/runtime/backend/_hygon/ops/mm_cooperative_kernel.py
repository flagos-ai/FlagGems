"""Producer/consumer GEMM over two ordinary shared-memory windows."""

from flag_gems.utils import libentry
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import mfma, buffer_store
from triton.experimental.gluon.language.hcu import (
    structured_register_load,
    matrix_load_to_shared,
    cta_barrier,
    warp_group_id,
    wait_local_memory,
)
from triton.experimental.gluon.language.amd.cdna4.async_copy import (
    commit_group,
    wait_group,
)
from .mm_shared_layout import (
    cooperative_shared_layout,
    shared_frame_layout,
    grouped_program_coordinates,
)

# Keep the full accumulator cohort across the staggered read/MMA boundary.
ACC_CONSTRAINTS = gl.constexpr(
    ",".join(["=v"] * 128 + [str(i) for i in range(128)] + ["~{memory}"])
)


@g.jit
def accumulator_fence(acc, PRIORITY: gl.constexpr):
    if PRIORITY == 1:
        acc = gl.inline_asm_elementwise(
            "s_setprio 1",
            ACC_CONSTRAINTS,
            [acc],
            dtype=gl.float32,
            is_pure=False,
            pack=128,
        )
    else:
        acc = gl.inline_asm_elementwise(
            "s_setprio 0",
            ACC_CONSTRAINTS,
            [acc],
            dtype=gl.float32,
            is_pure=False,
            pack=128,
        )
    return acc


@g.jit
def program_coordinates(M: gl.constexpr, N: gl.constexpr, GROUP_M: gl.constexpr):
    if GROUP_M == 0:
        return (gl.program_id(0), gl.program_id(1))
    else:
        count_m: gl.constexpr = (M + 255) // 256
        count_n: gl.constexpr = (N + 255) // 256
        return grouped_program_coordinates(
            gl.program_id(0) + gl.program_id(1) * count_m, count_m, count_n, GROUP_M
        )


@g.jit
def load_a64_parts(
    A,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    column,
    da: gl.constexpr,
    GROUP_M: gl.constexpr,
):
    pid_m, pid_n = program_coordinates(M, N, GROUP_M)
    full = structured_register_load(
        A, (M, K), (K, 1), (pid_m * 256, column, 1), (256, 64), da
    )
    lo, hi = gl.split(full.reshape((256, 2, 32)).permute((0, 2, 1)))
    return (gl.convert_layout(lo, da), gl.convert_layout(hi, da))


@g.jit
def producer_one(
    B,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    t,
    window,
    GROUP_M: gl.constexpr,
):
    pid_m, pid_n = program_coordinates(M, N, GROUP_M)
    dest = window._reinterpret(
        B.dtype.element_ty, [256, 64], cooperative_shared_layout(False)
    )
    matrix_load_to_shared(dest, B, (N, K), (K, 1), (pid_n * 256, t * 64))
    commit_group()
    # First phase retires reads of the other window; second publishes this DMA.
    cta_barrier()
    wait_group(0)
    cta_barrier()


@g.jit
def producer(
    B, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, owner, GROUP_M: gl.constexpr
):
    for pair in range(K // 128):
        producer_one(B, M, N, K, pair * 2, owner.index(0), GROUP_M)
        producer_one(B, M, N, K, pair * 2 + 1, owner.index(1), GROUP_M)
    if K // 64 % 2:
        producer_one(B, M, N, K, K // 64 - 1, owner.index(0), GROUP_M)
    # Distinct phases match the other actor even when adjacent in this one.
    cta_barrier()
    cta_barrier()


@g.jit
def compute_one(
    A,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    t,
    window,
    acc,
    av0,
    av1,
    high,
    GROUP_M: gl.constexpr,
):
    mma: gl.constexpr = AMDMFMALayout(
        3, [16, 16, 16], True, [8, 1], tiles_per_warp=[2, 2], mmac_layout=1
    )
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    parent = window._reinterpret(
        A.dtype.element_ty, [64, 256], cooperative_shared_layout(True)
    )
    bv0 = parent.slice(0, 32, dim=0).load(db)
    wait_local_memory()
    if high:
        cta_barrier()
    acc = accumulator_fence(acc, 1)
    acc = mfma(av0, bv0, acc)
    acc = accumulator_fence(acc, 0)
    if not high:
        cta_barrier()
    next0, next1 = load_a64_parts(A, M, N, K, (t + 1) * 64, da, GROUP_M)
    bv1 = parent.slice(32, 32, dim=0).load(db)
    wait_local_memory()
    if high:
        cta_barrier()
    acc = accumulator_fence(acc, 1)
    acc = mfma(av1, bv1, acc)
    acc = accumulator_fence(acc, 0)
    if not high:
        cta_barrier()
    return (acc, next0, next1)


@g.jit
def consumer(
    A,
    C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    owner,
    GROUP_M: gl.constexpr,
):
    pid_m, pid_n = program_coordinates(M, N, GROUP_M)
    mma: gl.constexpr = AMDMFMALayout(
        3, [16, 16, 16], True, [8, 1], tiles_per_warp=[2, 2], mmac_layout=1
    )
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    acc = gl.full((256, 256), 0, gl.float32, mma)
    av0, av1 = load_a64_parts(A, M, N, K, 0, da, GROUP_M)
    high = warp_group_id(4) != 0
    # Distinct phases match the other actor even when adjacent in this one.
    cta_barrier()
    cta_barrier()
    for pair in range(K // 128):
        acc, av0, av1 = compute_one(
            A, M, N, K, pair * 2, owner.index(0), acc, av0, av1, high, GROUP_M
        )
        acc, av0, av1 = compute_one(
            A, M, N, K, pair * 2 + 1, owner.index(1), acc, av0, av1, high, GROUP_M
        )
    if K // 64 % 2:
        acc, av0, av1 = compute_one(
            A, M, N, K, K // 64 - 1, owner.index(0), acc, av0, av1, high, GROUP_M
        )
    m = pid_m * 256 + gl.arange(0, 256, layout=gl.SliceLayout(1, mma))
    n = pid_n * 256 + gl.arange(0, 256, layout=gl.SliceLayout(0, mma))
    origin = pid_m.to(gl.int64) * 256 * N + pid_n.to(gl.int64) * 256
    cbase = C + origin
    local_m = gl.arange(0, 256, layout=gl.SliceLayout(1, mma))
    local_n = gl.arange(0, 256, layout=gl.SliceLayout(0, mma))
    offsets = local_m[:, None] * N + local_n[None, :]
    value = acc.to(C.dtype.element_ty)
    buffer_store(value, cbase, offsets, (m[:, None] < M) & (n[None, :] < N))


@libentry()
@g.jit
def _mm_cooperative_kernel(
    A,
    B,
    C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    GROUP_M: gl.constexpr = 0,
):
    gl.static_assert(K < 8192 and 512 * N < 2147483648)
    gl.static_assert(GROUP_M >= 0)
    gl.static_assert(K >= 64 and K % 64 == 0)
    gl.static_assert(M > 0 and N > 0 and (M < 2147483648) and (N < 2147483648))
    owner = gl.allocate_shared_memory(
        A.dtype.element_ty, [2, 16384], shared_frame_layout(16384)
    )
    gl.warp_specialize(
        [
            (consumer, (A, C, M, N, K, owner, GROUP_M)),
            (producer, (B, M, N, K, owner, GROUP_M)),
        ],
        [4],
        [32],
    )
    owner._keep_alive()
