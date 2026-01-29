import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


def supports_tma():
    return torch.cuda.get_device_capability()[0] >= 9


@triton.jit
def grouped_launch(
    pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


def get_autotune_config(pre_hook=None):
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
            num_ctas=1,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
            num_ctas=1,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
            num_ctas=2,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
            num_ctas=1,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
            num_ctas=2,
            pre_hook=pre_hook,
        ),
    ]


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def grouped_gemm_tma_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.bfloat16))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gk, gn],
                strides=[ldb, 1],
                block_shape=[BLOCK_K, BLOCK_N],
            )

            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, gm, gn, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_K])
                    b = b_desc.load([kk * BLOCK_K, offs_bn])
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                ori_c = c_desc.load([offs_cm, offs_cn])
                accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(c_desc.dtype)
                out_desc.store([offs_cm, offs_cn], c)

                tile_idx += total_grid

        last_problem_end = current_problem_end


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def grouped_gemm_kernel(
    M,
    N,
    K,
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_out_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    total_grid = tl.num_programs(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles
        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.bfloat16))

            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, gm, gn, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N

                a_ptrs = tl.make_block_ptr(
                    base=a_ptr,
                    shape=(gm, gk),
                    strides=(lda, 1),
                    offsets=(offs_am, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )
                b_ptrs = tl.make_block_ptr(
                    base=b_ptr,
                    shape=(gk, gn),
                    strides=(ldb, 1),
                    offsets=(0, offs_bn),
                    block_shape=(BLOCK_K, BLOCK_N),
                    order=(1, 0),
                )

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(gk, BLOCK_K)):
                    a = tl.load(a_ptrs, boundary_check=(0, 1))
                    b = tl.load(b_ptrs, boundary_check=(0, 1))
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
                    a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
                    b_ptrs = tl.advance(b_ptrs, (BLOCK_K, 0))

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                c_ptrs = tl.make_block_ptr(
                    base=c_ptr,
                    shape=(gm, gn),
                    strides=(ldc, 1),
                    offsets=(offs_cm, offs_cn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )

                out_ptrs = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(gm, gn),
                    strides=(ldc, 1),
                    offsets=(offs_cm, offs_cn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                ori_c = tl.load(c_ptrs, boundary_check=(0, 1))
                accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(c_ptrs.dtype.element_ty)
                tl.store(out_ptrs, c, boundary_check=(0, 1))

                tile_idx += total_grid

        last_problem_end = current_problem_end


@libentry()
@libtuner(
    configs=get_autotune_config(matmul_tma_set_block_size_hook), key=["M", "N", "K"]
)
@triton.jit
def grouped_mm_tma_kernel(
    a_desc,
    b_desc,
    c_desc,
    C,
    offs,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0
    group_end = 0

    for group_idx in tl.range(num_groups):
        group_end = tl.load(offs + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, m, N, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = group_start + tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N
                offs_bk = group_idx * K

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    a = a_desc.load([offs_am, k * BLOCK_K])
                    b = b_desc.load([offs_bk + k * BLOCK_K, offs_bn])
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

                c = accumulator.to(c_desc.dtype)

                if offs_am + BLOCK_M <= group_end:
                    c_desc.store([offs_am, offs_bn], c)
                else:
                    offs_cm = offs_am + tl.arange(0, BLOCK_M)
                    offs_cn = offs_bn + tl.arange(0, BLOCK_N)
                    c_ptrs = (
                        C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                    )
                    c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
                    tl.store(c_ptrs, c, mask=c_mask)

                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def grouped_mm_kernel(
    A,
    B,
    C,
    offs,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    total_grid = tl.num_programs(axis=0)
    tile_idx = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0
    group_end = 0

    for group_idx in tl.range(num_groups):
        group_end = tl.load(offs + group_idx).to(tl.int32)
        m = group_end - group_start
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        current_problem_end = last_problem_end + num_tiles
        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx, tile_n_idx = grouped_launch(
                    tile_idx_in_gemm, m, N, BLOCK_M, BLOCK_N, GROUP_M
                )

                offs_am = group_start + tile_m_idx * BLOCK_M
                offs_bn = tile_n_idx * BLOCK_N
                offs_bk = group_idx * K

                a_block_ptr = tl.make_block_ptr(
                    base=A,
                    shape=(M, K),
                    strides=(stride_am, stride_ak),
                    offsets=(offs_am, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )

                b_block_ptr = tl.make_block_ptr(
                    base=B,
                    shape=(num_groups * K, N),
                    strides=(stride_bk, stride_bn),
                    offsets=(offs_bk, offs_bn),
                    block_shape=(BLOCK_K, BLOCK_N),
                    order=(1, 0),
                )

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                    a = tl.load(a_block_ptr, boundary_check=(0, 1))
                    b = tl.load(b_block_ptr, boundary_check=(0, 1))
                    accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

                    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

                c = accumulator.to(C.dtype.element_ty)

                c_block_ptr = tl.make_block_ptr(
                    base=C,
                    shape=(M, N),
                    strides=(stride_cm, stride_cn),
                    offsets=(offs_am, offs_bn),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )

                if offs_am + BLOCK_M <= group_end:
                    tl.store(c_block_ptr, c, boundary_check=(0, 1))
                else:
                    offs_cm = offs_am + tl.arange(0, BLOCK_M)
                    offs_cn = offs_bn + tl.arange(0, BLOCK_N)
                    c_ptrs = (
                        C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                    )
                    c_mask = (offs_cm[:, None] < group_end) & (offs_cn[None, :] < N)
                    tl.store(c_ptrs, c, mask=c_mask)

                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end


def group_gemm(group_A, group_B, group_C, offs_table, alpha=1, beta=0):
    A_addrs = []
    B_addrs = []
    C_addrs = []
    group_sizes = []
    group_lds = []
    group_size = len(offs_table)
    M, N = group_C.shape
    K = group_A.shape[1]
    group_out = torch.empty((M, N), device=group_A.device, dtype=group_A.dtype)
    out_addrs = []
    for i in range(group_size):
        M_g = offs_table[i][0]
        N_g = offs_table[i][1]
        K_g = offs_table[i][2]
        A_g = group_A[offs_table[i][3]]
        B_g = group_B[offs_table[i][4]]
        C_g = group_C[offs_table[i][5]]
        out_g = group_out[offs_table[i][5]]
        group_sizes += [M_g, N_g, K_g]
        group_lds += [K_g, N_g, N_g]
        A_addrs.append(A_g.data_ptr())
        B_addrs.append(B_g.data_ptr())
        C_addrs.append(C_g.data_ptr())
        out_addrs.append(out_g.data_ptr())

    d_a_ptrs = torch.tensor(A_addrs, device=group_A.device)
    d_b_ptrs = torch.tensor(B_addrs, device=group_A.device)
    d_c_ptrs = torch.tensor(C_addrs, device=group_A.device)
    d_output_ptrs = torch.tensor(out_addrs, device=group_A.device)
    d_g_sizes = torch.tensor(group_sizes, dtype=torch.int32, device=group_A.device)
    d_g_lds = torch.tensor(group_lds, dtype=torch.int32, device=group_A.device)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    if hasattr(tl, "make_tensor_descriptor") and supports_tma():

        def alloc_fn(size, alignment, stream):
            return torch.empty(size, device=group_A.device, dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        grouped_gemm_tma_kernel[(NUM_SMS,)](
            M,
            N,
            K,
            d_a_ptrs,
            d_b_ptrs,
            d_c_ptrs,
            d_output_ptrs,
            d_g_sizes,
            d_g_lds,
            group_size,
            alpha=alpha,
            beta=beta,
        )
    else:
        grouped_gemm_kernel[(NUM_SMS,)](
            M,
            N,
            K,
            d_a_ptrs,
            d_b_ptrs,
            d_c_ptrs,
            d_output_ptrs,
            d_g_sizes,
            d_g_lds,
            group_size,
            alpha=alpha,
            beta=beta,
        )

    return group_out


def group_mm(A: torch.Tensor, B: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape

    num_groups, BK, N = B.shape
    strideBK, strideBN = B.stride(1), B.stride(2)

    assert num_groups == offs.numel()
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    C = A.new_empty(M, N)
    if hasattr(triton.tools.tensor_descriptor, "TensorDescriptor") and supports_tma():
        dummy_block = [1, 1]

        from triton.tools.tensor_descriptor import TensorDescriptor

        a_desc = TensorDescriptor(A, A.shape, A.stride(), dummy_block)
        b_desc = TensorDescriptor(
            B, [num_groups * K, N], [strideBK, strideBN], dummy_block
        )
        c_desc = TensorDescriptor(C, C.shape, C.stride(), dummy_block)

        grouped_mm_tma_kernel[(NUM_SMS,)](
            a_desc,
            b_desc,
            c_desc,
            C,
            offs,
            num_groups,
            M,
            N,
            K,
            C.stride(0),
            C.stride(1),
        )
    else:
        grouped_mm_kernel[(NUM_SMS,)](
            A,
            B,
            C,
            offs,
            num_groups,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            strideBK,
            strideBN,
            C.stride(0),
            C.stride(1),
        )

    return C
