import logging
import random

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


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


def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_M": GM},
            num_stages=s,
            num_warps=w,
        )
        for GM in [8]
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for BK in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def grouped_matmul_tma_kernel(
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
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
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
            while (
                tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles
            ):
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
                    accumulator = tl.dot(a, b, acc=accumulator)

                offs_cm = tile_m_idx * BLOCK_M
                offs_cn = tile_n_idx * BLOCK_N

                ori_c = c_desc.load([offs_cm, offs_cn])
                accumulator = ori_c * beta + accumulator * alpha

                c = accumulator.to(c_desc.dtype)
                out_desc.store([offs_cm, offs_cn], c)

                tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def grouped_matmul_kernel(
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
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            out_ptr = tl.load(group_out_ptrs + g).to(tl.pointer_type(tl.bfloat16))

            while (
                tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles
            ):
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
                    a = tl.load(a_ptrs, boundary_check=(1,))
                    b = tl.load(b_ptrs, boundary_check=(0,))
                    accumulator = tl.dot(a, b, acc=accumulator)
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

                tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


def supports_tma():
    return torch.cuda.get_device_capability()[0] >= 9


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
    grid = lambda META: (META["NUM_SM"],)

    if hasattr(tl, "make_tensor_descriptor") and supports_tma():

        def alloc_fn(size, alignment, stream):
            return torch.empty(size, device=group_A.device, dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        grouped_matmul_tma_kernel[grid](
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
            NUM_SM=NUM_SMS,
            alpha=alpha,
            beta=beta,
        )
    else:
        grouped_matmul_kernel[grid](
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
            NUM_SM=NUM_SMS,
            alpha=alpha,
            beta=beta,
        )

    return group_out


@pytest.mark.parametrize(
    "groups, N, K", [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_group_gemm_speedup(groups, N, K, dtype):
    M = [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128,
        136,
        144,
        152,
        160,
        168,
        176,
        184,
        192,
        200,
        208,
        216,
        224,
        232,
        240,
        248,
        256,
        272,
        288,
        304,
        320,
        336,
        352,
        368,
        384,
        400,
        416,
        432,
        448,
        464,
        480,
        496,
        512,
    ]
    group_A_list = []
    group_B_list = []
    group_C_list = []
    offs_table = []
    group_size = groups
    A_offs = 0
    B_offs = 0
    C_offs = 0
    alpha = 1
    beta = 1
    total_flops = 0

    for i in range(group_size):
        M_g = random.choice(M)
        N_g = N
        K_g = K
        A_g = torch.rand([M_g, K_g], device="cuda", dtype=torch.bfloat16)
        B_g = torch.rand([K_g, N_g], device="cuda", dtype=torch.bfloat16)
        C_g = torch.rand([M_g, N_g], device="cuda", dtype=torch.bfloat16)
        group_A_list.append(A_g)
        group_B_list.append(B_g)
        group_C_list.append(C_g)
        offs_table.append([M_g, N_g, K_g, A_offs, B_offs, C_offs])
        A_offs += M_g
        B_offs += K_g
        C_offs += M_g
        total_flops += 2 * M_g * N_g * K_g

    group_A = torch.cat([x for x in group_A_list], dim=0)
    group_B = torch.cat([x for x in group_B_list], dim=0)
    group_C = torch.cat([x for x in group_C_list], dim=0)

    latency = triton.testing.do_bench(
        lambda: group_gemm(group_A, group_B, group_C, offs_table, alpha, beta),
        warmup=1000,
        rep=100,
        return_mode="median",
    )

    def torch_addmm_fn():
        with flag_gems.use_gems():
            return [
                torch.addmm(
                    group_C_list[i],
                    group_A_list[i],
                    group_B_list[i],
                    alpha=alpha,
                    beta=beta,
                )
                for i in range(group_size)
            ]

    latency_base = triton.testing.do_bench(
        torch_addmm_fn, warmup=1000, rep=100, return_mode="median"
    )
    tflops = total_flops / latency / 1e12 * 1e3
    speedup = latency_base / latency

    latency_base_str = f"{latency_base:.6f}"
    latency_str = f"{latency:.6f}"
    speedup_str = f"{speedup:.3f}"
    tflops_str = f"{tflops:.3f}"
    col_names = [
        f"{'Base Latency (ms)'}",
        f"{'Gems Latency (ms)':>20}",
        f"{'Gems Speedup':>20}",
        f"{'TFLOPS':>20}",
    ]
    col_names_str = " ".join(col_names)
    print("\n\n" + col_names_str)
    header_break = "-" * len(col_names_str)
    print(header_break)
    print(
        f"{latency_base_str:>15}",
        f"{latency_str:>20}",
        f"{speedup_str:>20}",
        f"{tflops_str:>20}\n",
    )


@pytest.mark.parametrize(
    "groups, N, K", [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_group_gemm_accuracy(groups, N, K, dtype):
    M = [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128,
        136,
        144,
        152,
        160,
        168,
        176,
        184,
        192,
        200,
        208,
        216,
        224,
        232,
        240,
        248,
        256,
        272,
        288,
        304,
        320,
        336,
        352,
        368,
        384,
        400,
        416,
        432,
        448,
        464,
        480,
        496,
        512,
    ]
    group_A_list = []
    group_B_list = []
    group_C_list = []
    offs_table = []
    group_size = groups
    A_offs = 0
    B_offs = 0
    C_offs = 0
    alpha = 1
    beta = 1

    for i in range(group_size):
        M_g = random.choice(M)
        N_g = N
        K_g = K
        A_g = torch.rand([M_g, K_g], device="cuda", dtype=torch.bfloat16)
        B_g = torch.rand([K_g, N_g], device="cuda", dtype=torch.bfloat16)
        C_g = torch.rand([M_g, N_g], device="cuda", dtype=torch.bfloat16)
        group_A_list.append(A_g)
        group_B_list.append(B_g)
        group_C_list.append(C_g)
        offs_table.append([M_g, N_g, K_g, A_offs, B_offs, C_offs])
        A_offs += M_g
        B_offs += K_g
        C_offs += M_g

    group_A = torch.cat([x for x in group_A_list], dim=0)
    group_B = torch.cat([x for x in group_B_list], dim=0)
    group_C = torch.cat([x for x in group_C_list], dim=0)

    res = group_gemm(group_A, group_B, group_C, offs_table, alpha, beta)

    with flag_gems.use_gems():
        ref_out = [
            torch.addmm(
                group_C_list[i],
                group_A_list[i],
                group_B_list[i],
                alpha=alpha,
                beta=beta,
            )
            for i in range(group_size)
        ]
    ref = torch.cat([x for x in ref_out], dim=0)
    assert torch.allclose(res, ref, atol=1e-2, rtol=0)
