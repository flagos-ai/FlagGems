import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_spacemit"),
    key=["M", "N", "K"],
)
@triton.jit
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_M: tl.constexpr,
    SPLIT_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
    SUB_BLK_N: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    SUB_BLK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    if EVEN_K:
        a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
        a = smt.view(
            a_descriptor_load, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K)
        )
        b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
        b = smt.view(
            b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
        )
        accumulator = smt.dot(a, b)
        accumulator = smt.view(
            accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1)
        )
        c = accumulator.to(c_ptr.dtype.element_ty)
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            order=[1, 0],
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_M:
        b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
        b = smt.view(
            b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
        )
        sub_num = (
            min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1
        ) // SUB_BLK_M
        for s in smt.parallel(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(
                a_descriptor_load,
                (s * SUB_BLK_M, 0),
                (SUB_BLK_M, BLOCK_SIZE_K),
                (MICRO_M, MICRO_K),
            )
            accumulator = smt.dot(a, b)
            accumulator = smt.view(
                accumulator, (0, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1)
            )
            c = accumulator.to(c_ptr.dtype.element_ty)
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[pid_m * BLOCK_SIZE_M + s * SUB_BLK_M, pid_n * BLOCK_SIZE_N],
                block_shape=[SUB_BLK_M, BLOCK_SIZE_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_N:
        sub_num_m = (
            min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1
        ) // SUB_BLK_M
        sub_num_n = (
            min(BLOCK_SIZE_N, N - BLOCK_SIZE_N * pid_n) + SUB_BLK_N - 1
        ) // SUB_BLK_N
        total_sub_blocks = sub_num_m * sub_num_n
        b_alloc_ptr = smt.alloc(shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
        b_alloc_view_ptr = smt.view(
            b_alloc_ptr, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
        )
        bar = smt.mbarrier(flag=0, expect_count=sub_num_n)
        for s in smt.parallel(0, total_sub_blocks):
            s_m = s // sub_num_n
            s_n = s % sub_num_n
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(
                a_descriptor_load,
                (s_m * SUB_BLK_M, 0),
                (SUB_BLK_M, BLOCK_SIZE_K),
                (MICRO_M, MICRO_K),
            )
            b_alloc_sub_ptr = smt.view(
                b_alloc_view_ptr, (0, s_n * SUB_BLK_N), (BLOCK_SIZE_K, SUB_BLK_N)
            )
            if s_m == 0:
                b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
                b = smt.view(
                    b_descriptor_load,
                    (0, s_n * SUB_BLK_N),
                    (BLOCK_SIZE_K, SUB_BLK_N),
                    (MICRO_K, MICRO_N),
                )
                tl.store(b_alloc_sub_ptr, b, boundary_check=(0, 1, 2, 3))
                smt.barrier_arrive(bar)
            else:
                smt.barrier_wait(bar, flag=1)

            b_alloc = tl.load(b_alloc_sub_ptr, boundary_check=(0, 1, 2, 3))
            accumulator = smt.dot(a, b_alloc)
            accumulator = smt.view(accumulator, (0, 0), (SUB_BLK_M, SUB_BLK_N), (1, 1))
            c = accumulator.to(c_ptr.dtype.element_ty)
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[
                    pid_m * BLOCK_SIZE_M + s_m * SUB_BLK_M,
                    pid_n * BLOCK_SIZE_N + s_n * SUB_BLK_N,
                ],
                block_shape=[SUB_BLK_M, SUB_BLK_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_K:
        accumulator = tl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty
        )
        accumulator = smt.view(
            accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (MICRO_M, MICRO_N)
        )
        sub_num = (K + SUB_BLK_K - 1) // SUB_BLK_K
        for k in tl.range(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(
                a_descriptor_load,
                (0, k * SUB_BLK_K),
                (BLOCK_SIZE_M, SUB_BLK_K),
                (MICRO_M, MICRO_K),
            )
            b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
            b = smt.view(
                b_descriptor_load,
                (k * SUB_BLK_K, 0),
                (SUB_BLK_K, BLOCK_SIZE_N),
                (MICRO_K, MICRO_N),
            )
            accumulator += smt.dot(a, b)
        accumulator = smt.view(
            accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1)
        )
        c = accumulator.to(c_ptr.dtype.element_ty)

        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            order=[1, 0],
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))


def mm(a, b):
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    BLOCK_SIZE_K = triton.next_power_of_2(K)
    SUB_BLK_K = min(512, BLOCK_SIZE_K)

    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SUB_BLK_K=SUB_BLK_K,
    )
    return c
