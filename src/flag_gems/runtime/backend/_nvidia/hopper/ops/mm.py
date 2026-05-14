import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm_streamk import streamk_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.device_info import get_device_capability, get_sm_count
from flag_gems.utils.triton_version_utils import HAS_TLE, HAS_TLE_DEVICE_MESH

logger = logging.getLogger("flag_gems.runtime.backend._nvidia.hopper.ops.mm")
CACHE_USAGE_THRESHOLD = 0.8
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_hopper_expand.yaml")
)
_SHARED_MEM_SAFETY_MARGIN_BYTES = 1024


def _get_shared_memory_limit_bytes():
    """Return per-block opt-in shared-memory limit for current CUDA device."""
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).shared_memory_per_block_optin
    except Exception:
        return None


def _estimate_tma_shared_memory_bytes(block_m, block_n, block_k, num_stages):
    bytes_per_element = 4
    tile_bytes = (block_m * block_k + block_k * block_n) * bytes_per_element
    return tile_bytes * num_stages + _SHARED_MEM_SAFETY_MARGIN_BYTES


if HAS_TLE_DEVICE_MESH:
    import triton.experimental.tle.language as tle_exp

    BLOCK_CLUSTER_MESH = tle_exp.device_mesh({"block_cluster": [("cluster_x", 2)]})
    TLE_CLUSTER_SIZE = 2
    TLE_REMOTE_BM = 64
    TLE_REMOTE_BN = 256
    TLE_REMOTE_BK = 64
    TLE_REMOTE_NUM_WARPS = 8
    TLE_REMOTE_NUM_STAGES = 2
    TLE_REMOTE_A_SLOTS = 2
else:
    tle_exp = None
    BLOCK_CLUSTER_MESH = None
    TLE_CLUSTER_SIZE = 2
    TLE_REMOTE_BM = 64
    TLE_REMOTE_BN = 256
    TLE_REMOTE_BK = 64
    TLE_REMOTE_NUM_WARPS = 8
    TLE_REMOTE_NUM_STAGES = 2
    TLE_REMOTE_A_SLOTS = 2


def is_tma_compatible(a, b, N, K):
    """
    Check if tensors are compatible with TMA (Tensor Memory Accelerator).

    TMA requires 128-bit (16-byte) alignment for memory access:
    - For FP16/BF16 (2 bytes/element): N and K must be multiples of 8
      (8 elements × 2 bytes = 16 bytes)
    - For FP32 (4 bytes/element): N and K must be multiples of 4
      (4 elements × 4 bytes = 16 bytes)

    Args:
        a, b: Input tensors
        N, K: Matrix dimensions

    Returns:
        bool: True if compatible with TMA's alignment requirements
    """
    return (
        a.dtype in (torch.float16, torch.bfloat16)
        and b.dtype in (torch.float16, torch.bfloat16)
        and N % 8 == 0
        and K % 8 == 0
    ) or (
        a.dtype in (torch.float32,)
        and b.dtype in (torch.float32,)
        and N % 4 == 0
        and K % 4 == 0
    )


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    if nargs["A_ROW_MAJOR"]:
        nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M]

    if nargs["B_ROW_MAJOR"]:
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    else:
        nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]

    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["default", "default", "default", "default", "default"],
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

    if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0:
        # offset
        offset_am = pid_m * BLOCK_M
        offset_bn = pid_n * BLOCK_N
        offset_k = 0

        a_desc = tl.make_tensor_descriptor(
            base=A,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        # row-major
        b_desc = tl.make_tensor_descriptor(
            base=B,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_K, BLOCK_N],
        )

        # column-major
        # b_desc = tl.make_tensor_descriptor(
        #     B,
        #     shape = [N, K],
        #     strides = [K, 1],
        #     block_shape = [BLOCK_N, BLOCK_K],
        # )

        c_desc = tl.make_tensor_descriptor(
            base=C,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        if IS_FP64:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([offset_am.to(tl.int32), offset_k.to(tl.int32)])
            b = b_desc.load([offset_k.to(tl.int32), offset_bn.to(tl.int32)])
            if IS_FP64:
                acc += tl.dot(a, b, allow_tf32=False)
            else:
                acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            offset_k += BLOCK_K

        acc = acc.to(a_desc.dtype)
        c_desc.store([offset_am.to(tl.int32), offset_bn.to(tl.int32)], acc)

    else:
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
        offsets = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # handles write-back with reduction-splitting
        tl.store(offsets, acc, mask=mask)


def matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook):
    configs = [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [32, 64, 128, 256]
        for BN in [32, 64, 128]
        for BK in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
    shared_mem_limit = _get_shared_memory_limit_bytes()
    if shared_mem_limit is None:
        return configs

    filtered_configs = [
        cfg
        for cfg in configs
        if _estimate_tma_shared_memory_bytes(
            cfg.kwargs["BLOCK_M"],
            cfg.kwargs["BLOCK_N"],
            cfg.kwargs["BLOCK_K"],
            cfg.num_stages,
        )
        <= shared_mem_limit
    ]
    if not filtered_configs:
        logger.warning(
            "No mm_general_tma config fits shared memory limit (%s bytes); falling back to unfiltered configs.",
            shared_mem_limit,
        )
        return configs
    return filtered_configs


@libentry()
@libtuner(
    configs=runtime.ops_get_configs(
        "mm_general_tma",
        pre_hook=matmul_tma_set_block_size_hook,
        yaml_path=EXPAND_CONFIG_FILENAME,
    )
    if os.environ.get("USE_FLAGTUNE") == "1"
    else matmul_get_configs(),
    key=["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    strategy=runtime.get_expand_config(
        "mm_general_tma", yaml_path=EXPAND_CONFIG_FILENAME
    )["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
)
@triton.jit
def mm_kernel_general_host_tma(
    a_desc,
    b_desc,
    c_desc,
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
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
    dtype: tl.constexpr,
    enable_warp_specialization=True,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offset_am = (pid_m * BLOCK_M).to(tl.int32)
    offset_bn = (pid_n * BLOCK_N).to(tl.int32)
    iters = tl.cdiv(K, BLOCK_K)
    for k in range(iters):
        offset_ak = (k * BLOCK_K).to(tl.int32)

        if A_ROW_MAJOR:
            a = a_desc.load([offset_am, offset_ak])
        else:
            a_t = a_desc.load([offset_ak, offset_am])
            a = tl.trans(a_t)

        if B_ROW_MAJOR:
            b = b_desc.load([offset_ak, offset_bn])
        else:
            b_t = b_desc.load([offset_bn, offset_ak])
            b = tl.trans(b_t)

        if a_desc.dtype == tl.float16 or a_desc.dtype == tl.bfloat16:
            accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
        else:
            accumulator = tl.dot(a, b, acc=accumulator, input_precision="tf32x3")

    c = accumulator.to(c_desc.dtype)
    c_desc.store([offset_am, offset_bn], c)


def get_higher_dtype(a, b):
    _ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]

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
    # TODO: Remove this debug message
    logger.debug(
        "GEMS MM-hopper, [mm scenario]: general, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    # Broadcast tensors from expand() have stride=0, incompatible with TMA
    if 0 in a.stride():
        a = a.contiguous()
    if 0 in b.stride():
        b = b.contiguous()
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    if hasattr(
        triton.tools.tensor_descriptor, "TensorDescriptor"
    ) and is_tma_compatible(a, b, N, K):
        a_row_major = a.stride(1) == 1
        b_row_major = b.stride(1) == 1
        dummy_block = [1, 1]
        # triton 3.5.0
        from triton.tools.tensor_descriptor import TensorDescriptor

        if a_row_major:
            a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
        else:
            a_desc = TensorDescriptor(a, a.T.shape, a.T.stride(), dummy_block)
        if b_row_major:
            b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
        else:
            b_desc = TensorDescriptor(b, b.T.shape, b.T.stride(), dummy_block)
        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

        input_dtype = a.dtype
        dtype_str = str(input_dtype).split(".")[-1]

        with torch_device_fn.device(a.device):
            mm_kernel_general_host_tma[grid](
                a_desc,
                b_desc,
                c_desc,
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
                A_ROW_MAJOR=a_row_major,
                B_ROW_MAJOR=b_row_major,
                dtype=dtype_str,
            )
    else:

        def alloc_fn(size: int, align: int, stream: Optional[int]):
            return torch.empty(size, dtype=torch.int8, device=a.device)

        triton.set_allocator(alloc_fn)

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
    configs=runtime.ops_get_configs(
        "gemv", pre_hook=None, yaml_path=EXPAND_CONFIG_FILENAME
    )
    if os.environ.get("USE_FLAGTUNE") == "1"
    else [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 256},
        )
    ],
    key=["M", "K", "stride_am", "stride_bk"],
    strategy=runtime.get_expand_config("gemv", yaml_path=EXPAND_CONFIG_FILENAME)[
        "strategy"
    ]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def gemv_kernel(
    A,
    B,
    C,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    """Optimized kernel for matrix-vector multiplication (N=1 case)"""
    pid = tl.program_id(0)

    # Each program handles BLOCK_M rows
    row_start = pid * BLOCK_M
    row_offset = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offset < M

    # Accumulator for this block of rows
    if IS_FP64:
        acc = tl.zeros((BLOCK_M,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offset = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        # Load block from matrix A: [BLOCK_M, BLOCK_K]
        a_ptrs = A + row_offset[:, None] * stride_am + k_offset[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Load block from vector B: [BLOCK_K]
        b_ptrs = B + k_offset * stride_bk
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        # Accumulate: sum over K dimension
        if IS_FP64:
            acc += tl.sum(a * b[None, :], axis=1)
        else:
            acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)

    # Store result
    c_ptrs = C + row_offset
    acc = acc.to(C.dtype.element_ty)
    tl.store(c_ptrs, acc, mask=row_mask)


def gemv_mm(a, b, c, M, K):
    """Optimized matrix-vector multiplication for N=1 case"""
    logger.debug(
        "GEMS MM-hopper, [mm scenario]: gemv (N=1), [shape info]: [%s, %s, 1](M, K, N)",
        M,
        K,
    )

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch_device_fn.device(a.device):
        gemv_kernel[grid](
            a,
            b,
            c,
            M,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            IS_FP64=a.dtype == torch.float64,
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
) -> tuple:
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

    # Optimize for N=1 case (matrix-vector multiplication)
    if N == 1:
        return gemv_mm(a, b, c, M, K)
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, sm_count=sm_count)
    if HAS_TLE and BLOCK_CLUSTER_MESH is not None:
        if cluster_remote_mm_scenario(a, b, c, M, N, K):
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

    # Optimize for N=1 case (matrix-vector multiplication)
    if N == 1:
        return gemv_mm(a, b, out, M, K)
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, out, M, N, K, sm_count=sm_count)
    if HAS_TLE and BLOCK_CLUSTER_MESH is not None:
        if cluster_remote_mm_scenario(a, b, out, M, N, K):
            return cluster_remote_mm(a, b, out, M, N, K)
    return general_mm(a, b, out, M, N, K)
