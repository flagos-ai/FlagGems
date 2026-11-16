# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import wraps

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm_streamk import streamk_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.device_info import get_device_capability, get_sm_count
from flag_gems.utils.triton_version_utils import (  # noqa: F401
    HAS_TLE,
    HAS_TLE_DEVICE_MESH,
    _triton_version_at_least,
)

if HAS_TLE_DEVICE_MESH:
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
    IS_FP64: tl.constexpr = False,
):
    # matrix multiplication
    pid = ext.program_id(0)
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

    if IS_FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if IS_FP64:
            acc += tl.dot(a, b, allow_tf32=False)
        else:
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
    if IS_FP64:
        acc += tl.dot(a, b, allow_tf32=False)
    else:
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
    use_mask = (
        (M % bm != 0)
        or (N % bn != 0)
        or (K % bk != 0)
        or (triton.cdiv(N, bn) % TLE_CLUSTER_SIZE != 0)
    )
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


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


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
            IS_FP64=a.dtype == torch.float64,
        )
    return c


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_self_transpose"),
    key=["M", "K", "stride_am", "stride_ak"],
    strategy=["align32", "align32", "align32", "align32"],
    warmup=2,
    rep=4,
)
@triton.jit
def mm_kernel_syrk(
    A,
    C,
    M,
    K,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    # Packed lower-triangular launch domain:
    #   pid = row * (row + 1) / 2 + col, where 0 <= col <= row.
    #
    # Invert the triangular-number indexing by solving:
    #   row^2 + row - 2 * pid = 0
    # => row = (-1 + sqrt(1 + 8 * pid)) / 2
    #
    # We take floor(...) as the candidate row, then apply an integer +/-1 correction
    # because fp32 sqrt can be off near triangular-number boundaries.
    pid_f = pid.to(tl.float32)
    pid_m = tl.floor((tl.sqrt(8.0 * pid_f + 1.0) - 1.0) / 2.0).to(tl.int32)
    tri_start = pid_m * (pid_m + 1) // 2
    pid_m = tl.where(tri_start > pid, pid_m - 1, pid_m)
    next_tri_start = (pid_m + 1) * (pid_m + 2) // 2
    pid_m = tl.where(next_tri_start <= pid, pid_m + 1, pid_m)
    tri_start = pid_m * (pid_m + 1) // 2
    pid_n = pid - tri_start

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    ran = tl.max_contiguous(tl.multiple_of(rn % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K
        a = tl.load(
            A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            A + (rk[:, None] * stride_ak + ran[None, :] * stride_am),
            mask=mask_k[:, None],
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    out = acc.to(C.dtype.element_ty)
    c_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < M)[None, :]
    tl.store(c_ptr, out, mask=mask)

    if pid_m > pid_n:
        c_t_ptr = C + (rn[:, None] * stride_cm + rm[None, :] * stride_cn)
        mask_t = (rn < M)[:, None] & (rm < M)[None, :]
        tl.store(c_t_ptr, tl.trans(out), mask=mask_t)


def is_syrk_transpose_pair(a, b):
    return (
        a.ndim == 2
        and b.ndim == 2
        and a.shape[0] == b.shape[1]
        and a.shape[1] == b.shape[0]
        and a.stride(0) == b.stride(1)
        and a.stride(1) == b.stride(0)
        and a.storage_offset() == b.storage_offset()
        and a.data_ptr() == b.data_ptr()
    )


def syrk_mm(a, c, M, K):
    grid = lambda META: (
        # Number of tile rows is tiles = ceil(M / BLOCK_M).
        # Packed lower triangle contains:
        #   1 + 2 + ... + tiles = tiles * (tiles + 1) / 2
        triton.cdiv(M, META["BLOCK_M"])
        * (triton.cdiv(M, META["BLOCK_M"]) + 1)
        // 2,
    )
    with torch_device_fn.device(a.device):
        mm_kernel_syrk[grid](
            a,
            c,
            M,
            K,
            a.stride(0),
            a.stride(1),
            c.stride(0),
            c.stride(1),
        )
    return c


def streamk_scenario(a, b, M, N, K):
    # TODO: this my change sometime according to the realbenchmark result
    # Currently, the best configuration for streamk has only been tested on A100(capability[0] == 8).
    # The optimal settings for other devices need to be determined through real testing.
    #
    # kunlunxin (XPU): the STREAM-K kernels are not usable here. Their scalar
    # spin-lock atomics (`locks + tile_id`, tile_id derived via integer
    # division) fail to lower in TritonSDNNBufferize:
    #   error: unexpected op in ptr sequence
    #   note: Pipeline failed while executing [TritonSDNNBufferize ...]
    # so the kernel never compiles (RuntimeError: PassManager::run failed).
    # XPU also reports capability (8, 6), which is what selects this path.
    if runtime.device.vendor_name == "kunlunxin":
        return False
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
    if mm_needs_autograd(a, b):
        return MmFunction.apply(a, b)
    logger.debug("GEMS MM")

    device = a.device
    if a.numel() == 0 or b.numel() == 0:
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        return torch.zeros(
            (a.shape[0], b.shape[1]),
            device=device,
            dtype=get_higher_dtype(a.dtype, b.dtype),
        )
    if is_syrk_transpose_pair(a, b):
        M, K = a.shape
        c = torch.empty((M, M), device=device, dtype=a.dtype)
        return syrk_mm(a, c, M, K)
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
        return cluster_remote_mm(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)


def mm_needs_autograd(a, b):
    """Whether the public ``mm`` entry must build a first-order autograd node."""
    return bool((a.requires_grad or b.requires_grad) and torch.is_grad_enabled())


def mm_out(a, b, *, out):
    logger.debug("GEMS MM_OUT")

    if is_syrk_transpose_pair(a, b):
        M, K = a.shape
        return syrk_mm(a, out, M, K)
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
        return cluster_remote_mm(a, b, out, M, N, K)
    return general_mm(a, b, out, M, N, K)


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM for MoE router gate. weight shape: (N, K)."""
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)
    b = weight.t().contiguous()
    return general_mm(x, b, c, M, N, K)


@triton.jit
def _mm_tile(
    A,
    B,
    C,
    pid,
    rows,
    cols,
    red,
    sa_r,
    sa_k,
    sb_k,
    sb_c,
    sc_r,
    sc_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32X3: tl.constexpr,
):
    """Write one BLOCK_M x BLOCK_N tile of ``C = A @ B``.

    ``A`` is described by ``(sa_r, sa_k)`` and ``B`` by ``(sb_k, sb_c)``, so a
    transposed view of either operand is contracted correctly without a copy.
    """
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(cols, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_r = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_r[:, None] * sa_r + offs_k[None, :] * sa_k
    b_ptrs = B + offs_k[:, None] * sb_k + offs_c[None, :] * sb_c
    r_mask = offs_r < rows
    c_mask = offs_c < cols

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, red, BLOCK_K):
        # offs_k is a fixed arange, so the tail mask must add the block start;
        # otherwise a reduction length that is not a multiple of BLOCK_K would
        # silently drop the last partial block.
        k_mask = (k_start + offs_k) < red
        a = tl.load(a_ptrs, mask=r_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & c_mask[None, :], other=0.0)
        if TF32X3:
            # Same policy as the forward mm kernels: IEEE-accurate FP32 on the
            # tensor cores via the three-pass TF32 decomposition.
            acc = tl.dot(a, b, acc, input_precision="tf32x3")
        else:
            acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_k

    c_ptrs = C + offs_r[:, None] * sc_r + offs_c[None, :] * sc_c
    tl.store(
        c_ptrs,
        acc.to(C.dtype.element_ty),
        mask=r_mask[:, None] & c_mask[None, :],
    )


@triton.jit
def _mm_backward_kernel(
    GRAD_OUT,
    A,
    B,
    GRAD_A,
    GRAD_B,
    M,
    N,
    K,
    go_s0,
    go_s1,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    grad_a_tiles,
    NEED_A: tl.constexpr,
    NEED_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TF32X3: tl.constexpr,
):
    """Both gradients of ``out = A @ B`` in one launch.

    The first ``grad_a_tiles`` programs write ``grad_a`` and the rest write
    ``grad_b``.  A gradient that was not requested has an empty tile range and
    its branch is compiled out, so the launch always computes exactly the
    requested set.
    """
    pid = tl.program_id(0)
    if NEED_A:
        if pid < grad_a_tiles:
            # grad_a = grad_output @ B.T: contract over N, B is read as (N, K).
            _mm_tile(
                GRAD_OUT,
                B,
                GRAD_A,
                pid,
                M,
                K,
                N,
                go_s0,
                go_s1,
                b_s1,
                b_s0,
                K,
                1,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                TF32X3,
            )
        else:
            if NEED_B:
                # grad_b = A.T @ grad_output: contract over M, A is read as
                # (K, M); output row k is A's second axis, hence stride N/1.
                _mm_tile(
                    A,
                    GRAD_OUT,
                    GRAD_B,
                    pid - grad_a_tiles,
                    K,
                    N,
                    M,
                    a_s1,
                    a_s0,
                    go_s0,
                    go_s1,
                    N,
                    1,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    GROUP_M,
                    TF32X3,
                )
    else:
        if NEED_B:
            _mm_tile(
                A,
                GRAD_OUT,
                GRAD_B,
                pid,
                K,
                N,
                M,
                a_s1,
                a_s0,
                go_s0,
                go_s1,
                N,
                1,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                TF32X3,
            )


# Below this many multiply-accumulates one fused launch beats two tuned
# forward launches, because the forward entry spends ~0.1 ms per call on host
# dispatcher work while a plain Triton launch costs ~0.01 ms.  The limits are
# measured on H800 with the core shapes of benchmark/test_mm.py: 384^3 and
# 1024^3 are host-bound for every dtype, 2048^3 is still host-bound for the
# 16-bit dtypes, and 4096^3 is GPU-bound for all of them.  FP32 crosses over
# earlier because the fused tile performs three TF32 passes per product, which
# only pays off while the launch itself is the bottleneck.  Both kernels are
# correct on either side of the limit, so another backend may move the
# constant without changing semantics.
_HOST_BOUND_LIMIT = {
    torch.float32: 2_000_000_000,
    torch.float16: 20_000_000_000,
    torch.bfloat16: 20_000_000_000,
}

_GROUP_M = 8


def _tile_shape(dtype, rows, cols):
    """Fused-kernel tile, from the measured per-shape sweep on H800.

    Narrow tiles for narrow problems (a 128-wide tile on a 15-row matrix is
    mostly padding), wider tiles once there is enough work to fill the GPU.
    """
    narrow = rows <= 512 and cols <= 512
    if dtype == torch.float32:
        if narrow:
            return 32, 32, 32, 4, 3
        return 128, 64, 32, 4, 3
    if narrow:
        return 64, 64, 64, 4, 3
    if rows * cols <= 4_000_000:
        return 128, 128, 64, 8, 4
    return 128, 256, 64, 8, 3


def _needs_fused_launch(dtype, m, n, k):
    limit = _HOST_BOUND_LIMIT.get(dtype, 20_000_000_000)
    return m * n * k < limit


def _fused_mm_backward(grad_output, a, b, grad_a, grad_b, need_a, need_b):
    m, k = a.shape
    n = b.shape[1]
    dtype = a.dtype
    # grad_a's tile grid is (M, K) and grad_b's is (K, N); both branches share
    # one compile-time configuration, chosen from the wider of the two grids.
    rows = max(m if need_a else 0, k if need_b else 0)
    cols = k if need_a else 0
    if need_b:
        cols = max(cols, n)
    block_m, block_n, block_k, warps, stages = _tile_shape(dtype, rows, cols)
    tiles_a = triton.cdiv(m, block_m) * triton.cdiv(k, block_n) if need_a else 0
    tiles_b = triton.cdiv(k, block_m) * triton.cdiv(n, block_n) if need_b else 0
    if tiles_a + tiles_b == 0:  # pragma: no cover - guarded by mm_backward
        return
    with torch_device_fn.device(a.device):
        _mm_backward_kernel[(tiles_a + tiles_b,)](
            grad_output,
            a,
            b,
            grad_a if need_a else grad_output,
            grad_b if need_b else grad_output,
            m,
            n,
            k,
            grad_output.stride(0),
            grad_output.stride(1),
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            tiles_a,
            NEED_A=need_a,
            NEED_B=need_b,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=_GROUP_M,
            TF32X3=dtype == torch.float32,
            num_warps=warps,
            num_stages=stages,
        )


def _tuned_mm(x, y):
    """One GEMM through the tuned public entry (no autograd node)."""
    from flag_gems import mm as gemm

    return gemm(x.detach(), y.detach())


def mm_backward(grad_output, a, b, needs_input_grad):
    """First-order gradients of ``torch.mm`` for real 2-D operands.

    ``a`` is ``(M, K)``, ``b`` is ``(K, N)`` and ``grad_output`` is ``(M, N)``.
    ``needs_input_grad`` mirrors ``ctx.needs_input_grad``: an entry that is
    ``False`` returns ``None`` instead of launching that GEMM.

    Zero-length dimensions are answered by allocation only: ``M == 0`` or
    ``K == 0`` makes ``grad_a`` empty, and a zero-length contraction makes the
    corresponding gradient an empty or all-zero tensor, exactly as ATen does.
    """
    logger.debug("GEMS MM_BACKWARD")
    if grad_output.is_complex() or a.is_complex() or b.is_complex():
        raise NotImplementedError(
            "FlagGems mm backward does not support complex inputs"
        )
    m, k = a.shape
    n = b.shape[1]
    need_a = bool(needs_input_grad[0])
    need_b = bool(needs_input_grad[1])
    if not need_a and not need_b:
        return None, None

    # A zero-length output dimension is empty and a contraction over no terms
    # is zero; both are answered by allocation, without a kernel launch.
    grad_a = None
    if need_a:
        if m != 0 and k != 0 and n == 0:
            grad_a = torch.zeros((m, k), dtype=a.dtype, device=a.device)
        else:
            grad_a = torch.empty((m, k), dtype=a.dtype, device=a.device)

    grad_b = None
    if need_b:
        if k != 0 and n != 0 and m == 0:
            grad_b = torch.zeros((k, n), dtype=b.dtype, device=b.device)
        else:
            grad_b = torch.empty((k, n), dtype=b.dtype, device=b.device)

    if m == 0 or k == 0 or n == 0:
        return grad_a, grad_b

    if _needs_fused_launch(a.dtype, m, n, k):
        _fused_mm_backward(grad_output, a, b, grad_a, grad_b, need_a, need_b)
    else:
        if need_a:
            grad_a = _tuned_mm(grad_output, b.transpose(0, 1)).to(a.dtype)
        if need_b:
            grad_b = _tuned_mm(a.transpose(0, 1), grad_output).to(b.dtype)
    return grad_a, grad_b


class MmFunction(torch.autograd.Function):
    """First-order autograd node around the generic ``mm`` forward."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return mm(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return mm_backward(grad_output, a, b, ctx.needs_input_grad)


def _with_hopper_mm_backward(forward):
    """Attach the generic gradient kernels to the selected Hopper forward."""

    def run_forward(a, b):
        if a.shape[1] == b.shape[0] == 0:
            return torch.zeros(
                (a.shape[0], b.shape[1]),
                dtype=get_higher_dtype(a.dtype, b.dtype),
                device=a.device,
            )
        return forward(a, b)

    class HopperMmFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return run_forward(a, b)

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            return mm_backward(grad_output, a, b, ctx.needs_input_grad)

    @wraps(forward)
    def mm_with_backward(a, b):
        if mm_needs_autograd(a, b):
            return HopperMmFunction.apply(a, b)
        return run_forward(a, b)

    return mm_with_backward
