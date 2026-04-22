import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm_streamk import streamk_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.device_info import get_device_capability, get_sm_count
from flag_gems.utils.triton_version_utils import (  # noqa: F401
    HAS_TLE,
    _triton_version_at_least,
)

if HAS_TLE:
    import triton.experimental.tle.language as tle_exp

    BLOCK_CLUSTER_MESH = tle_exp.device_mesh({"block_cluster": [("cluster_x", 2)]})
else:
    tle_exp = None
    BLOCK_CLUSTER_MESH = None

CACHE_USAGE_THRESHOLD = 0.8
TLE_CLUSTER_SIZE = 2
TLE_REMOTE_BM = 64
TLE_REMOTE_BN = 256
TLE_REMOTE_BK = 64
TLE_REMOTE_NUM_WARPS = 8
TLE_REMOTE_NUM_STAGES = 2
TLE_REMOTE_A_SLOTS = 2

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def mm_kernel_general(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    prev_multiple = prev_multiple_of(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # loop peeling
    rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
        mask=mask_k[None, :],
        other=0.0,
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
        mask=mask_k[:, None],
        other=0.0,
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


if HAS_TLE:

    @triton.jit
    def _cluster_remote_gemm_kernel(
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
        mesh: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        DOT_K: tl.constexpr,
        CLUSTER_SIZE: tl.constexpr,
        USE_MASK: tl.constexpr,
        A_SLOTS: tl.constexpr,
        USE_NV_MMA_SMEM_LAYOUT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cluster_rank = tle_exp.shard_id(mesh, "cluster_x")
        cluster_id = pid // CLUSTER_SIZE

        num_pid_n = tl.cdiv(N, BN)
        num_pid_n_group = tl.cdiv(num_pid_n, CLUSTER_SIZE)
        pid_m = cluster_id // num_pid_n_group
        pid_ng = cluster_id % num_pid_n_group
        pid_n = pid_ng * CLUSTER_SIZE + cluster_rank

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_row_base = offs_m - pid_m * BM
        a_rows_full = tl.broadcast_to(a_row_base[:, None], (BM, BK))
        a_cols_full = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
        a_rows_t = tl.broadcast_to(a_row_base[None, :], (DOT_K, BM))
        a_buf = tle_exp.gpu.alloc(
            [A_SLOTS, BM, BK],
            dtype=tl.float16,
            layout=None,
            scope=tle_exp.gpu.smem,
            nv_mma_shared_layout=USE_NV_MMA_SMEM_LAYOUT,
        )
        a_buf_remote = tle_exp.remote(a_buf, 0, scope=mesh)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        slot0 = 0
        slot0_full = tl.zeros((BM, BK), dtype=tl.int32) + slot0
        if cluster_rank == 0:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            if USE_MASK:
                a_mask_tile = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                a_tile = tl.load(a_ptrs, mask=a_mask_tile, other=0.0)
            else:
                a_tile = tl.load(a_ptrs)
            a_local_ptr_tile = tle_exp.gpu.local_ptr(
                a_buf, (slot0_full, a_rows_full, a_cols_full)
            )
            if USE_MASK:
                tl.store(a_local_ptr_tile, a_tile, mask=a_mask_tile)
            else:
                tl.store(a_local_ptr_tile, a_tile)

        tle_exp.distributed_barrier(mesh)

        for k0 in range(0, K, BK):
            iter_idx = k0 // BK
            slot = iter_idx % A_SLOTS

            for ks in range(0, BK, DOT_K):
                k_local = ks + tl.arange(0, DOT_K)
                a_cols_t = tl.broadcast_to(k_local[:, None], (DOT_K, BM))
                slot_dot_t = tl.zeros((DOT_K, BM), dtype=tl.int32) + slot
                a_ptr_remote = tle_exp.gpu.local_ptr(
                    a_buf_remote, (slot_dot_t, a_rows_t, a_cols_t)
                )
                if USE_MASK:
                    a_mask_t = ((k0 + k_local)[:, None] < K) & (offs_m[None, :] < M)
                    a = tl.trans(tl.load(a_ptr_remote, mask=a_mask_t, other=0.0))
                else:
                    a = tl.trans(tl.load(a_ptr_remote))

                b_ptrs = (
                    b_ptr
                    + (k0 + k_local)[:, None] * stride_bk
                    + offs_n[None, :] * stride_bn
                )
                if USE_MASK:
                    b_mask = ((k0 + k_local)[:, None] < K) & (offs_n[None, :] < N)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                else:
                    b = tl.load(b_ptrs)
                acc = tl.dot(a, b, acc)

            if A_SLOTS == 1:
                tle_exp.distributed_barrier(mesh)

            next_k0 = k0 + BK
            has_next = next_k0 < K
            next_iter = iter_idx + 1
            next_slot = next_iter % A_SLOTS
            next_slot_full = tl.zeros((BM, BK), dtype=tl.int32) + next_slot
            if has_next and cluster_rank == 0:
                a_ptrs = (
                    a_ptr
                    + offs_m[:, None] * stride_am
                    + (next_k0 + offs_k)[None, :] * stride_ak
                )
                if USE_MASK:
                    a_mask_tile = (offs_m[:, None] < M) & (
                        (next_k0 + offs_k)[None, :] < K
                    )
                    a_tile = tl.load(a_ptrs, mask=a_mask_tile, other=0.0)
                else:
                    a_tile = tl.load(a_ptrs)
                a_local_ptr_tile = tle_exp.gpu.local_ptr(
                    a_buf, (next_slot_full, a_rows_full, a_cols_full)
                )
                if USE_MASK:
                    tl.store(a_local_ptr_tile, a_tile, mask=a_mask_tile)
                else:
                    tl.store(a_local_ptr_tile, a_tile)

            tle_exp.distributed_barrier(mesh)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        if USE_MASK:
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
        else:
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def _select_remote_dot_k(bk: int) -> int:
    if bk % 16 == 0:
        return 16
    raise ValueError(f"BK must be divisible by 16 for remote dot path, got BK={bk}")


def _grid_cluster_remote(
    M: int,
    N: int,
    BM: int,
    BN: int,
    cluster_size: int = TLE_CLUSTER_SIZE,
) -> tuple[int]:
    num_pid_n = triton.cdiv(N, BN)
    num_pid_n_group = triton.cdiv(num_pid_n, cluster_size)
    return (triton.cdiv(M, BM) * num_pid_n_group,)


def _run_cluster_remote(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    bm: int,
    bn: int,
    bk: int,
    num_warps: int,
    num_stages: int,
) -> None:
    M, K = a.shape
    N = b.shape[1]
    dot_k = _select_remote_dot_k(bk)
    use_mask = (M % bm != 0) or (N % bn != 0) or (K % bk != 0)
    a_slots = TLE_REMOTE_A_SLOTS
    use_nv_mma_smem_layout = (bk == 32) or (bk == 64 and num_stages <= 2)
    _cluster_remote_gemm_kernel[_grid_cluster_remote(M, N, bm, bn)](
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
        mesh=BLOCK_CLUSTER_MESH,
        BM=bm,
        BN=bn,
        BK=bk,
        DOT_K=dot_k,
        CLUSTER_SIZE=TLE_CLUSTER_SIZE,
        USE_MASK=use_mask,
        A_SLOTS=a_slots,
        USE_NV_MMA_SMEM_LAYOUT=use_nv_mma_smem_layout,
        num_ctas=1,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def cluster_remote_mm_scenario(a, b, c, M, N, K):
    capability = get_device_capability()
    return (
        HAS_TLE
        and BLOCK_CLUSTER_MESH is not None
        and capability[0] >= 9
        and a.is_cuda
        and b.is_cuda
        and c.is_cuda
        and a.dtype == torch.float16
        and b.dtype == torch.float16
        and c.dtype == torch.float16
        and a.is_contiguous()
        and b.is_contiguous()
        and M >= TLE_REMOTE_BM
        and N >= TLE_REMOTE_BN
        and K >= TLE_REMOTE_BK
    )


def cluster_remote_mm(a, b, c, M, N, K):
    logger.debug(
        "GEMS MM [cluster_remote]: M=%s N=%s K=%s, A_col_major=%s, B_col_major=%s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    with torch_device_fn.device(a.device):
        _run_cluster_remote(
            a,
            b,
            c,
            TLE_REMOTE_BM,
            TLE_REMOTE_BN,
            TLE_REMOTE_BK,
            TLE_REMOTE_NUM_WARPS,
            TLE_REMOTE_NUM_STAGES,
        )
    return c


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def general_mm(a, b, c, M, N, K):
    logger.debug(
        "GEMS MM, [mm scenario]: general, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_general[grid](
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
            GROUP_M=8,
        )
    return c


def streamk_scenario(a, b, M, N, K):
    # TODO: this my change sometime according to the realbenchmark result
    # Currently, the best configuration for streamk has only been tested on A100(capability[0] == 8).
    # The optimal settings for other devices need to be determined through real testing.
    capability = get_device_capability()
    return (
        capability[0] == 8
        and a.dtype in [torch.float16, torch.bfloat16]
        and b.dtype in [torch.float16, torch.bfloat16]
        and a.is_contiguous()
        and b.is_contiguous()
        and K > M * 5
        and K > N * 5
    )


def mm(a, b):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, sm_count=sm_count)
    if cluster_remote_mm_scenario(a, b, c, M, N, K):
        print("走了tle分支")
        return cluster_remote_mm(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, out, M, N, K, sm_count=sm_count)
    if cluster_remote_mm_scenario(a, b, out, M, N, K):
        print("走了tle分支")
        return cluster_remote_mm(a, b, out, M, N, K)
    return general_mm(a, b, out, M, N, K)
