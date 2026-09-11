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

import functools
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
from flag_gems.utils.libentry import LibTuner
from flag_gems.utils.triton_version_utils import HAS_TLE, HAS_TLE_DEVICE_MESH

logger = logging.getLogger(__name__)
CACHE_USAGE_THRESHOLD = 0.8
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_hopper_expand.yaml")
)
_SHARED_MEM_SAFETY_MARGIN_BYTES = 1024
_TMA_TRANSPOSE_BLOCK_N = 64
_H20_SM_COUNT = 78
_H20_NEAR_WAVE_MIN_CTAS = 70


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


def _get_sm_count_for_tensor(tensor):
    try:
        if tensor.is_cuda:
            return torch.cuda.get_device_properties(tensor.device).multi_processor_count
    except Exception:
        pass
    return get_sm_count()


def _estimate_tma_shared_memory_bytes(
    block_m, block_n, block_k, num_stages, bytes_per_element=4
):
    tile_bytes = (block_m * block_k + block_k * block_n) * bytes_per_element
    return tile_bytes * num_stages + _SHARED_MEM_SAFETY_MARGIN_BYTES


tle_exp = None
tlc = None
tle_types = None
if HAS_TLE:
    try:
        import triton.experimental.tle.language as tle_exp
        import triton.language.core as tlc
        from triton.experimental.tle.language.gpu import types as tle_types
    except (ImportError, AttributeError):
        pass

HAS_TLE_WARP_SPECIALIZATION = (
    tle_exp is not None
    and tlc is not None
    and tle_types is not None
    and hasattr(tlc, "_unwrap_if_constexpr")
    and all(
        hasattr(tle_exp.gpu, name)
        for name in (
            "READY",
            "alloc",
            "alloc_barriers",
            "barrier_arrive",
            "barrier_wait",
            "copy",
            "smem",
            "warp_specialize",
            "wgmma",
            "wgmma_wait",
        )
    )
)


if HAS_TLE_DEVICE_MESH and tle_exp is not None:

    BLOCK_CLUSTER_MESH = tle_exp.device_mesh({"block_cluster": [("cluster_x", 2)]})
    TLE_CLUSTER_SIZE = 2
    TLE_REMOTE_BM = 64
    TLE_REMOTE_BN = 256
    TLE_REMOTE_BK = 64
    TLE_REMOTE_NUM_WARPS = 8
    TLE_REMOTE_NUM_STAGES = 2
    TLE_REMOTE_A_SLOTS = 2
else:
    BLOCK_CLUSTER_MESH = None
    TLE_CLUSTER_SIZE = 2
    TLE_REMOTE_BM = 64
    TLE_REMOTE_BN = 256
    TLE_REMOTE_BK = 64
    TLE_REMOTE_NUM_WARPS = 8
    TLE_REMOTE_NUM_STAGES = 2
    TLE_REMOTE_A_SLOTS = 2


def _is_tma_descriptor_aligned(tensor, allow_transpose=False):
    if tensor.ndim != 2 or tensor.data_ptr() % 16 != 0:
        return False
    layout = tensor
    if layout.stride(1) != 1:
        if not allow_transpose or layout.stride(0) != 1:
            return False
        layout = layout.T
    return layout.stride(0) * layout.element_size() % 16 == 0


def is_tma_compatible(a, b, N, K, c=None):
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
    dtype_and_shape_compatible = (
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
    return (
        dtype_and_shape_compatible
        and _is_tma_descriptor_aligned(a, allow_transpose=True)
        and _is_tma_descriptor_aligned(b, allow_transpose=True)
        and (c is None or _is_tma_descriptor_aligned(c))
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
    key=["M", "N", "K", "stride_am", "stride_bk", "USE_TMA"],
    strategy=["default", "default", "default", "default", "default", "default"],
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
    USE_TMA: tl.constexpr,
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

    if USE_TMA and M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0:
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
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_M": 8},
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
    configs += [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_M": GM},
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BN in [32, 64, 128]
        for BK in [64, 128, 256]
        for GM in [1, 8]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]
    configs += [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": 256, "GROUP_M": GM},
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [32, 64]
        for BN in [32, 64]
        for GM in [1, 8]
        for s in [2, 3]
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
            bytes_per_element=2,
        )
        <= shared_mem_limit
    ]
    if not filtered_configs:
        logger.warning(
            "GEMS_NVIDIA No mm_general_tma config fits shared memory limit (%s bytes); "
            "falling back to unfiltered configs.",
            shared_mem_limit,
        )
        return configs
    return filtered_configs


@libentry()
@libtuner(
    configs=matmul_get_configs(),
    key=["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    strategy=["align32", "align32", "align32", "align32", "align32", "default"],
    policy="flagtune",
    warmup=5,
    rep=5,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_general_tma",
    flagtune_op_id="flaggems/mm",
    flagtune_variant="general_tma",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=matmul_tma_set_block_size_hook,
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


def _sync_mm_host_tma_descriptor_block_shapes(args, kwargs):
    if len(args) < 3:
        return
    block_m = kwargs.get("BLOCK_M")
    block_n = kwargs.get("BLOCK_N")
    block_k = kwargs.get("BLOCK_K")
    a_row_major = kwargs.get("A_ROW_MAJOR")
    b_row_major = kwargs.get("B_ROW_MAJOR")
    if None in (block_m, block_n, block_k, a_row_major, b_row_major):
        return

    a_desc, b_desc, c_desc = args[:3]
    if not all(hasattr(desc, "block_shape") for desc in (a_desc, b_desc, c_desc)):
        return

    a_desc.block_shape = [block_m, block_k] if a_row_major else [block_k, block_m]
    b_desc.block_shape = [block_k, block_n] if b_row_major else [block_n, block_k]
    c_desc.block_shape = [block_m, block_n]


def _install_mm_host_tma_descriptor_block_shape_guard():
    jit_fn = mm_kernel_general_host_tma.fn.fn
    if getattr(jit_fn, "_flag_gems_mm_tma_block_shape_guard", False):
        return

    original_run = jit_fn.run

    def run_with_descriptor_block_shapes(*args, **kwargs):
        _sync_mm_host_tma_descriptor_block_shapes(args, kwargs)
        return original_run(*args, **kwargs)

    jit_fn.run = run_with_descriptor_block_shapes
    jit_fn._flag_gems_mm_tma_block_shape_guard = True


_install_mm_host_tma_descriptor_block_shape_guard()


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


def general_mm(a, b, c, M, N, K, op_name="mm"):
    # TODO: Remove this debug message
    logger.debug(
        "GEMS_NVIDIA MM_HOPPER, [op]: %s, [mm scenario]: general, [shape info]: "
        "[-, %s, %s, %s](batch, M, N, K), [A column-major]: %s, [B column-major]: %s",
        op_name,
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
    ) and is_tma_compatible(a, b, N, K, c):
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
                A_ROW_MAJOR=a_row_major,
                B_ROW_MAJOR=b_row_major,
                dtype=dtype_str,
            )
    else:

        def alloc_fn(size: int, align: int, stream: Optional[int]):
            return torch.empty(size, dtype=torch.int8, device=a.device)

        triton.set_allocator(alloc_fn)

        use_device_tma = (
            a.stride() == (K, 1)
            and b.stride() == (N, 1)
            and c.stride() == (N, 1)
            and _is_tma_descriptor_aligned(a)
            and _is_tma_descriptor_aligned(b)
            and _is_tma_descriptor_aligned(c)
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
                USE_TMA=use_device_tma,
                IS_FP64=a.dtype == torch.float64,
            )
    return c


@triton.jit
def mm_kernel_tma_transposed_splitk_partials(
    a_desc,
    b_desc,
    partials,
    M,
    N,
    K,
    stride_ps,
    stride_pm,
    stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    tile_id = tl.program_id(0)
    split_id = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = tile_id // grid_n
    pid_n = tile_id % grid_n

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    k_blocks = tl.cdiv(K, BLOCK_K)
    blocks_per_split = tl.cdiv(k_blocks, SPLIT_K)
    k_begin = split_id * blocks_per_split
    k_end = min(k_begin + blocks_per_split, k_blocks)

    accumulator_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k_block in range(k_begin, k_end):
        offset_k = (k_block * BLOCK_K).to(tl.int32)
        a = a_desc.load([offset_m, offset_k])
        if B_ROW_MAJOR:
            b_t = tl.trans(b_desc.load([offset_k, offset_n]))
        else:
            b_t = b_desc.load([offset_n, offset_k])
        accumulator_t = tl.dot(
            b_t,
            tl.trans(a),
            acc=accumulator_t,
            allow_tf32=False,
        )

    rows = offset_m + tl.arange(0, BLOCK_M)
    cols = offset_n + tl.arange(0, BLOCK_N)
    partial_ptrs = (
        partials
        + split_id * stride_ps
        + rows[:, None] * stride_pm
        + cols[None, :] * stride_pn
    )
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(partial_ptrs, tl.trans(accumulator_t), mask=mask)


@triton.jit
def mm_kernel_tma_transposed_direct(
    a_desc,
    b_desc,
    output,
    M,
    N,
    K,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    tile_id = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    if N_MAJOR:
        pid_m = tile_id % grid_m
        pid_n = tile_id // grid_m
    else:
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n
    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)

    accumulator_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = (k_block * BLOCK_K).to(tl.int32)
        a = a_desc.load([offset_m, offset_k])
        if B_ROW_MAJOR:
            b_t = tl.trans(b_desc.load([offset_k, offset_n]))
        else:
            b_t = b_desc.load([offset_n, offset_k])
        accumulator_t = tl.dot(
            b_t,
            tl.trans(a),
            acc=accumulator_t,
            allow_tf32=False,
        )

    rows = offset_m + tl.arange(0, BLOCK_M)
    cols = offset_n + tl.arange(0, BLOCK_N)
    output_ptrs = output + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(output_ptrs, tl.trans(accumulator_t), mask=mask)


def _set_tma_transposed_direct_block_shapes(
    a_desc, b_desc, block_m, block_n, block_k, b_row_major
):
    a_desc.block_shape = [block_m, block_k]
    b_desc.block_shape = [block_k, block_n] if b_row_major else [block_n, block_k]


def _tma_transposed_direct_set_block_size_hook(nargs):
    _set_tma_transposed_direct_block_shapes(
        nargs["a_desc"],
        nargs["b_desc"],
        nargs["BLOCK_M"],
        nargs["BLOCK_N"],
        nargs["BLOCK_K"],
        nargs["B_ROW_MAJOR"],
    )


def _get_tma_transposed_direct_tuned_configs():
    configs = runtime.get_tuned_config("mm_tma_transposed_direct")
    for config in configs:
        config.pre_hook = _tma_transposed_direct_set_block_size_hook
    return configs


class _TmaTransposedStableTuner(LibTuner):
    def policy(self, bench_fn, configs, args, kwargs):
        del args, kwargs
        timings = {config: bench_fn(config) for config in configs}
        p50 = {config: float(values[0]) for config, values in timings.items()}
        fastest = min(p50.values())
        near_ties = [config for config in timings if p50[config] <= fastest * 1.005]
        best_config = max(
            near_ties,
            key=lambda config: (
                config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"],
                config.kwargs["BLOCK_M"],
                -p50[config],
            ),
        )
        return best_config, timings


mm_kernel_tma_transposed_direct_tuned = libentry()(
    libtuner(
        configs=_get_tma_transposed_direct_tuned_configs(),
        key=["M", "N", "K", "stride_bk"],
        strategy=["default", "default", "default", "default"],
        policy=_TmaTransposedStableTuner,
        warmup=10,
        rep=20,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_tma_transposed_direct",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        flagtune_pre_hook=_tma_transposed_direct_set_block_size_hook,
    )(mm_kernel_tma_transposed_direct)
)


@triton.jit
def mm_kernel_tma_splitk_reduce(
    partials,
    output,
    n_elements,
    partial_stride,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split_id in range(SPLIT_K):
        accumulator += tl.load(
            partials + split_id * partial_stride + offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(output + offsets, accumulator, mask=mask)


def _tma_transposed_splitk_config(M, N, K):
    if M < 1 or N % 64 or K % 128:
        return None

    m_rows = (M + 15) // 16
    if m_rows > 3 or not (m_rows & 1):
        return None

    sm_count = _H20_SM_COUNT
    output_ctas = m_rows * (N // 64)
    output_waves = (output_ctas + sm_count - 1) // sm_count
    tail_ctas = output_ctas - (output_waves - 1) * sm_count
    k_steps = K // 128

    if not 8 * N <= 4 * K < 9 * N:
        return None

    split8_k_steps = (k_steps + 7) // 8
    if 2 * sm_count <= 8 * output_ctas <= 4 * sm_count and 3 <= split8_k_steps <= 5:
        return 16, 128, 8, 2, 4

    split4_k_steps = (k_steps + 3) // 4
    if (
        sm_count < output_ctas
        and 4 * output_ctas <= 6 * sm_count
        and tail_ctas <= sm_count // 4
        and 6 <= split4_k_steps <= 10
    ):
        return 16, 128, 4, 2, 4
    return None


def tma_transposed_config(a, b, c, M, N, K):
    config = _tma_transposed_splitk_config(M, N, K)
    if config is None or not _tma_transposed_runtime_eligible(a, b, c, M, N, K):
        return None

    return config


def _tma_transposed_runtime_eligible(a, b, c, M, N, K):
    descriptor_module = getattr(
        getattr(triton, "tools", None), "tensor_descriptor", None
    )
    descriptor_type = getattr(descriptor_module, "TensorDescriptor", None)
    if not (
        descriptor_type is not None
        and hasattr(descriptor_type, "from_tensor")
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and c.dtype == torch.bfloat16
        and a.is_cuda
        and b.is_cuda
        and c.is_cuda
        and a.device == b.device == c.device
        and tuple(c.shape) == (M, N)
        and a.is_contiguous()
        and (b.is_contiguous() or b.T.is_contiguous())
        and c.is_contiguous()
        and _is_tma_descriptor_aligned(a)
        and _is_tma_descriptor_aligned(b, allow_transpose=True)
        and _is_tma_descriptor_aligned(c)
    ):
        return False

    try:
        properties = torch.cuda.get_device_properties(a.device)
    except Exception:
        return False
    if properties.major != 9 or properties.multi_processor_count != _H20_SM_COUNT:
        return False
    return True


def _tile_row_band(extent, tile, first_row, last_row, first_tail=1, last_tail=None):
    lower = (first_row - 1) * tile + first_tail
    upper = last_row * tile
    if last_tail is not None:
        upper = (last_row - 1) * tile + last_tail
    return lower <= extent <= upper


def _tma_transposed_direct_tuned_shape(M, N, K):
    if M < 1 or N % 64 or K % 128:
        return False

    sm_count = _H20_SM_COUNT
    m_rows_16 = (M + 15) // 16
    m_rows_32 = (M + 31) // 32
    m_rows_64 = (M + 63) // 64
    m_tail_64 = M - (m_rows_64 - 1) * 64
    n_cols_64 = (N + 63) // 64
    n_cols_128 = (N + 127) // 128
    n_cols_160 = (N + 159) // 160
    n_cols_256 = (N + 255) // 256
    k_steps_128 = K // 128
    medium_k = 12 <= k_steps_128 <= 20

    if medium_k and 1 <= n_cols_64 <= 4:
        return m_rows_16 * n_cols_64 <= sm_count or (
            5 * K <= 32 * M and 4 * M <= K and m_rows_32 * n_cols_64 <= sm_count
        )

    half_ratio = 8 * N <= 4 * K < 9 * N

    if (
        half_ratio
        and k_steps_128 >= 16
        and _tile_row_band(M, 32, 9, 9)
        and (7 < n_cols_64 < 9 or 7 < n_cols_128 < 9 or 7 < n_cols_256 < 9)
    ):
        return True

    if half_ratio and medium_k:
        direct_ctas = m_rows_16 * n_cols_64
        direct_waves = (direct_ctas + sm_count - 1) // sm_count
        tail_ctas = direct_ctas - (direct_waves - 1) * sm_count
        return (
            direct_ctas <= sm_count
            or (
                m_rows_32 * n_cols_64 <= sm_count
                and _tile_row_band(M, 32, 3, 4, first_tail=8, last_tail=8)
            )
            or _tile_row_band(M, 16, 9, 14, first_tail=8)
            or (
                4 * M <= K
                and direct_ctas > 4 * sm_count
                and direct_waves & 1
                and 24 <= tail_ctas <= 40
            )
        )

    if half_ratio and 24 <= k_steps_128 <= 40:
        direct_ctas = m_rows_16 * n_cols_64
        direct_waves = (direct_ctas + sm_count - 1) // sm_count
        tail_ctas = direct_ctas - (direct_waves - 1) * sm_count
        return (
            _tile_row_band(M, 16, 2, 2)
            or _tile_row_band(M, 16, 5, 7, first_tail=8)
            or _tile_row_band(M, 16, 9, 11, first_tail=8, last_tail=8)
            or _tile_row_band(M, 32, 7, 7, first_tail=8)
            or (
                8 * sm_count < direct_ctas <= 9 * sm_count
                and tail_ctas >= sm_count // 2
            )
        )

    short_k = (
        3 <= k_steps_128 <= 5 and 24 <= n_cols_64 <= 40 and 16 * K <= 4 * N < 17 * K
    )
    if short_k:
        output_ctas = m_rows_32 * n_cols_64
        return (
            _tile_row_band(M, 16, 1, 3, first_tail=16)
            or _tile_row_band(M, 16, 5, 7, first_tail=8)
            or _tile_row_band(M, 16, 9, 11, first_tail=8, last_tail=8)
            or (7 <= m_rows_32 <= 19 and 2 * sm_count < output_ctas <= 8 * sm_count)
        )

    near_n_wave = (
        medium_k
        and _H20_NEAR_WAVE_MIN_CTAS <= n_cols_128 <= sm_count
        and 17 * K <= 4 * N < 20 * K
    )
    if near_n_wave:
        general_ctas = m_rows_64 * n_cols_128
        return (
            _tile_row_band(M, 16, 1, 3)
            or _tile_row_band(M, 32, 3, 3, first_tail=32)
            or _tile_row_band(M, 16, 7, 7)
            or (2 * sm_count < general_ctas <= 4 * sm_count and m_tail_64 <= 32)
            or (5 * sm_count < general_ctas <= 6 * sm_count and m_tail_64 <= 16)
        )

    fragmented_n_wave = (
        medium_k
        and n_cols_128 > sm_count
        and _H20_NEAR_WAVE_MIN_CTAS <= n_cols_160 <= sm_count
        and 20 * K <= 4 * N < 28 * K
    )
    if fragmented_n_wave:
        return (
            _tile_row_band(M, 8, 1, 1)
            or _tile_row_band(M, 16, 2, 3)
            or _tile_row_band(M, 64, 3, 3)
            or (4 <= m_rows_64 <= 7 and m_tail_64 <= 32)
        )

    return medium_k and n_cols_128 >= 16 * sm_count and _tile_row_band(M, 8, 1, 1)


def _tma_transposed_direct_column_major_shape(M, N, K):
    if M < 1 or N < 64 or N % 32 or K % 128:
        return False

    sm_count = _H20_SM_COUNT
    m_rows_32 = (M + 31) // 32
    n_cols_64 = (N + 63) // 64
    k_steps_128 = K // 128
    if k_steps_128 < 12:
        return False

    direct_ctas = m_rows_32 * n_cols_64
    if 4 * N >= 28 * K:
        return False

    if 4 * M <= K and 2 * direct_ctas <= 5 * sm_count:
        return True

    if M <= 96:
        return True

    if M <= 128 and K <= 8 * N and (4 * N < 12 * K or n_cols_64 >= 3 * sm_count):
        return True

    if 2 * N < K:
        return False

    if M <= 512 and direct_ctas <= 5 * sm_count:
        return True

    m_tail_64 = M - ((M - 1) // 64) * 64
    return m_rows_32 <= 5 and m_tail_64 <= 32


def _tma_transposed_direct_general_shape(M, N, K):
    if M < 1 or N < 32 or N % 32 or K < 256 or K % 128:
        return False
    if K < 1024 and N < 8 * K:
        return False

    m_rows_32 = (M + 31) // 32
    n_cols_64 = (N + 63) // 64
    direct_ctas = m_rows_32 * n_cols_64
    deep_wide = (
        32 <= K // 128 <= 64
        and m_rows_32 <= 4
        and 2 * _H20_SM_COUNT <= n_cols_64 <= 5 * _H20_SM_COUNT
    )
    return (K <= 4 * N and (direct_ctas <= 5 * _H20_SM_COUNT or deep_wide)) or (
        (m_rows_32 == 1 or 56 <= M <= 96) and n_cols_64 > 5 * _H20_SM_COUNT
    )


def tma_transposed_direct_tuned_scenario(a, b, c, M, N, K):
    shape_matches = _tma_transposed_direct_tuned_shape(M, N, K)
    if not shape_matches:
        column_major_shape = (
            not b.is_contiguous()
            and b.T.is_contiguous()
            and _tma_transposed_direct_column_major_shape(M, N, K)
        )
        shape_matches = (
            (_tma_transposed_direct_general_shape(M, N, K) or column_major_shape)
            and _tma_transposed_splitk_config(M, N, K) is None
            and not splitk_scenario(a, b, M, N, K)
        )
    return shape_matches and _tma_transposed_runtime_eligible(a, b, c, M, N, K)


def tma_transposed_direct_tuned_mm(a, b, c, M, N, K):
    logger.debug(
        "GEMS_NVIDIA MM_HOPPER, [mm scenario]: tuned transposed host-TMA, "
        "[shape info]: [-, %s, %s, %s](batch, M, N, K)",
        M,
        N,
        K,
    )
    from triton.tools.tensor_descriptor import TensorDescriptor

    dummy_block = [1, 1]
    b_row_major = b.stride(1) == 1
    a_desc = TensorDescriptor.from_tensor(a, block_shape=dummy_block)
    b_desc = TensorDescriptor.from_tensor(
        b if b_row_major else b.T, block_shape=dummy_block
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_tma_transposed_direct_tuned[grid](
            a_desc,
            b_desc,
            c,
            M,
            N,
            K,
            b.stride(0),
            c.stride(0),
            c.stride(1),
            B_ROW_MAJOR=b_row_major,
        )
    return c


def tma_transposed_mm(a, b, c, M, N, K, config, reduce_block=None):
    block_m, block_k, split_k, num_stages, num_warps = config
    block_n = _TMA_TRANSPOSE_BLOCK_N
    logger.debug(
        "GEMS_NVIDIA MM_HOPPER, [mm scenario]: transposed split-K host-TMA, "
        "[shape info]: [-, %s, %s, %s](batch, M, N, K), split_k=%s",
        M,
        N,
        K,
        split_k,
    )
    from triton.tools.tensor_descriptor import TensorDescriptor

    b_row_major = b.stride(1) == 1
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(
        b if b_row_major else b.T,
        block_shape=[block_k, block_n] if b_row_major else [block_n, block_k],
    )
    output_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)

    with torch_device_fn.device(a.device):
        partials = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
        mm_kernel_tma_transposed_splitk_partials[(output_tiles, split_k)](
            a_desc,
            b_desc,
            partials,
            M,
            N,
            K,
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=split_k,
            B_ROW_MAJOR=b_row_major,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if reduce_block is None:
            reduce_block = 256 if split_k == 8 else 512
        mm_kernel_tma_splitk_reduce[(triton.cdiv(M * N, reduce_block),)](
            partials,
            c,
            M * N,
            partials.stride(0),
            SPLIT_K=split_k,
            BLOCK_SIZE=reduce_block,
            num_warps=8,
        )
    return c


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"BLOCK_M": 1, "BLOCK_K": 128},
            num_warps=1,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 256},
        ),
    ],
    key=["M", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "default"],
    policy="flagtune",
    warmup=5,
    rep=10,
    flagtune_op_name="mm",
    flagtune_expand_op_name="gemv",
    flagtune_op_id="flaggems/mm",
    flagtune_variant="gemv",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=None,
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
    stride_bn,
    stride_cm,
    stride_cn,
    partial_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_VECTORS: tl.constexpr = 1,
    SPLIT_K: tl.constexpr = 1,
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

    if SPLIT_K == 1:
        block_start = 0
        block_end = tl.cdiv(K, BLOCK_K)
        split_id = 0
    else:
        split_id = tl.program_id(1)
        total_blocks = tl.cdiv(K, BLOCK_K)
        blocks_per_split = tl.cdiv(total_blocks, SPLIT_K)
        block_start = split_id * blocks_per_split
        block_end = min(block_start + blocks_per_split, total_blocks)
    if N_VECTORS == 1:
        vector_id = 0
    else:
        vector_id = tl.program_id(2)

    for block in range(block_start, block_end):
        k_offset = block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        # Load block from matrix A: [BLOCK_M, BLOCK_K]
        a_ptrs = A + row_offset[:, None] * stride_am + k_offset[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Load block from vector B: [BLOCK_K]
        b_ptrs = B + k_offset * stride_bk + vector_id * stride_bn
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        # Accumulate: sum over K dimension
        if IS_FP64:
            acc += tl.sum(a * b[None, :], axis=1)
        else:
            acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)

    # Store result
    c_ptrs = (
        C + split_id * partial_stride + row_offset * stride_cm + vector_id * stride_cn
    )
    acc = acc.to(C.dtype.element_ty)
    tl.store(c_ptrs, acc, mask=row_mask)


def gemv_mm(a, b, c, M, K):
    """Optimized matrix-vector multiplication for N=1 case"""
    logger.debug(
        "GEMS_NVIDIA MM_HOPPER, [mm scenario]: gemv (N=1), [shape info]: [%s, %s, 1](M, K, N)",
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
            b.stride(1),
            c.stride(0),
            c.stride(1),
            0,
            N_VECTORS=1,
            IS_FP64=a.dtype == torch.float64,
        )
    return c


def _splitk_gemv_runtime_eligible(a, b, c, K):
    return (
        K >= 16384
        and a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.is_cuda
        and b.is_cuda
        and c.is_cuda
        and a.device == b.device == c.device
        and a.is_contiguous()
        and (b.is_contiguous() or b.T.is_contiguous())
        and c.is_contiguous()
    )


def _splitk_gemv_scenario(a, b, c, M, N, K):
    return (
        M == 1
        and 1 < N <= 4
        and tuple(c.shape) == (M, N)
        and _splitk_gemv_runtime_eligible(a, b, c, K)
    )


def _batched_splitk_gemv_scenario(a, b, c, M, N, K):
    return (
        1 < M
        and 1 < N <= 4
        and M * N <= 16 * _get_sm_count_for_tensor(a)
        and tuple(c.shape) == (M, N)
        and _splitk_gemv_runtime_eligible(a, b, c, K)
    )


def splitk_gemv_mm(a, b, c, M, N, K):
    if M == 1:
        matrix = b.T
        vectors = a[0]
        output_rows = N
        output_vectors = 1
        if matrix.is_contiguous():
            block_m, block_k, split_k, num_warps = 1, 128, 32, 1
        else:
            block_m, block_k, split_k, num_warps = 4, 256, 64, 4
    else:
        matrix = a
        vectors = b
        output_rows = M
        output_vectors = N
        block_k = 256
        if M <= 16:
            block_m, split_k, num_warps = 16, 32, 4
        elif M <= 96:
            block_m, split_k, num_warps = 8, 16, 2
        else:
            block_m, split_k, num_warps = 32, 16, 2

    partials = torch.empty(
        (split_k, output_rows, output_vectors),
        device=a.device,
        dtype=torch.float32,
    )
    raw_gemv_kernel = gemv_kernel.fn.fn
    with torch_device_fn.device(a.device):
        raw_gemv_kernel[(triton.cdiv(output_rows, block_m), split_k, output_vectors)](
            matrix,
            vectors,
            partials,
            output_rows,
            K,
            matrix.stride(0),
            matrix.stride(1),
            vectors.stride(0),
            0 if vectors.ndim == 1 else vectors.stride(1),
            partials.stride(1),
            partials.stride(2),
            partials.stride(0),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            N_VECTORS=output_vectors,
            SPLIT_K=split_k,
            IS_FP64=False,
            num_warps=num_warps,
            num_stages=1,
        )
        n_elements = M * N
        reduce_block = 32 if M == 1 else 256
        mm_kernel_tma_splitk_reduce[(triton.cdiv(n_elements, reduce_block),)](
            partials,
            c,
            n_elements,
            partials.stride(0),
            SPLIT_K=split_k,
            BLOCK_SIZE=reduce_block,
            num_warps=1 if M == 1 else 4,
        )
    return c


@triton.jit
def _mm_kernel_splitk(
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
    stride_cs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    STORE_PARTIALS: tl.constexpr,
):
    tile_m: tl.constexpr = 16 if BLOCK_M < 16 else BLOCK_M
    tile_n: tl.constexpr = 16 if BLOCK_N < 16 else BLOCK_N
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_n = tl.cdiv(N, tile_n)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offset_am = pid_m * tile_m
    offset_bn = pid_n * tile_n
    offs_am = offset_am + tl.arange(0, tile_m)
    offs_bn = offset_bn + tl.arange(0, tile_n)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((tile_m, tile_n), dtype=tl.float32)
    for k in range(k_start, k_end):
        offset_k = k * BLOCK_K
        offs_k = offset_k + tl.arange(0, BLOCK_K)

        a = tl.load(
            A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    offs_cm = offset_am + tl.arange(0, tile_m)
    offs_cn = offset_bn + tl.arange(0, tile_n)
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    if STORE_PARTIALS:
        tl.store(c_ptrs + pid_k * stride_cs, acc, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)


mm_kernel_splitk = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("mm_splitk"),
        key=["M", "N", "K", "stride_am", "stride_bk"],
        reset_to_zero=["C"],
        strategy=["align32", "align32", "align32", "align32", "align32"],
        warmup=5,
        rep=10,
        policy="flagtune",
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_splitk",
        flagtune_op_id="flaggems/mm",
        flagtune_variant="splitk",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        flagtune_pre_hook=None,
    )(_mm_kernel_splitk)
)


def _prune_mm_splitk_two_step_configs(configs, named_args, **kwargs):
    del kwargs
    block_ns = (16, 32) if named_args["N"] <= 32 else (64, 128)
    pruned_configs = [
        config for config in configs if config.kwargs["BLOCK_N"] in block_ns
    ]
    return pruned_configs or list(configs)


mm_kernel_splitk_partials = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("mm_splitk_two_step"),
        key=["M", "N", "K", "stride_am", "stride_bk"],
        strategy=["default", "default", "default", "default", "default"],
        prune_configs_by={"early_config_prune": _prune_mm_splitk_two_step_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_splitk_two_step",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )(_mm_kernel_splitk)
)


@triton.jit
def mm_kernel_splitk_reduce_strided(
    partials,
    output,
    M,
    N,
    stride_cm,
    stride_cn,
    partial_stride,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split_id in range(SPLIT_K):
        accumulator += tl.load(
            partials + split_id * partial_stride + offsets,
            mask=mask,
            other=0.0,
        )
    offsets_m = offsets // N
    offsets_n = offsets % N
    output_ptrs = output + offsets_m * stride_cm + offsets_n * stride_cn
    tl.store(output_ptrs, accumulator, mask=mask)


_TWO_STEP_MAX_SPLITS = 32
_TWO_STEP_MIN_K_ITERS_PER_SPLIT = 2
_TWO_STEP_MULTI_WAVE_MIN_OUTPUT_TILES = 12


def _floor_power_of_two(value):
    if value < 1:
        return 0
    return 1 << (int(value).bit_length() - 1)


def _ceil_power_of_two(value):
    if value <= 1:
        return 1
    return 1 << int(value - 1).bit_length()


@functools.lru_cache(maxsize=1)
def _splitk_two_step_tile_candidates():
    candidates = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
        )
        for config in runtime.get_tuned_config("mm_splitk_two_step")
    }
    expand_config = runtime.get_expand_config(
        "mm_splitk_two_step", yaml_path=EXPAND_CONFIG_FILENAME
    )
    if expand_config != -1:
        ranges = expand_config["ranges"]
        candidates.update(
            (block_m, block_n, block_k)
            for block_m in ranges.get("BLOCK_M", ())
            for block_n in ranges.get("BLOCK_N", ())
            for block_k in ranges.get("BLOCK_K", ())
        )
    return tuple(sorted(candidates))


def _splitk_two_step_reference_tile(N):
    block_ns = (16, 32) if N <= 32 else (64, 128)
    candidates = tuple(
        candidate
        for candidate in _splitk_two_step_tile_candidates()
        if candidate[1] in block_ns
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda tile: (tile[0] * tile[1], tile[0], tile[1], tile[2]),
    )


def _splitk_two_step_split_k(a, M, N, K):
    reference_tile = _splitk_two_step_reference_tile(N)
    if reference_tile is None:
        return 1
    block_m, block_n, block_k = reference_tile
    output_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    k_iters = triton.cdiv(K, block_k)
    max_k_split = _floor_power_of_two(k_iters // _TWO_STEP_MIN_K_ITERS_PER_SPLIT)
    if N <= 32:
        if output_tiles <= 1:
            occupancy_split_k = 32
        elif output_tiles <= 16:
            occupancy_split_k = 16
        else:
            occupancy_split_k = 8
        return max(1, min(occupancy_split_k, max_k_split, _TWO_STEP_MAX_SPLITS))

    sm_count = _get_sm_count_for_tensor(a)
    if output_tiles < _TWO_STEP_MULTI_WAVE_MIN_OUTPUT_TILES:
        target_programs = max(1, 3 * sm_count // 4)
    else:
        target_programs = max(1, 3 * sm_count // 2)
    occupancy_split_k = _ceil_power_of_two(triton.cdiv(target_programs, output_tiles))
    return max(1, min(occupancy_split_k, max_k_split, _TWO_STEP_MAX_SPLITS))


def _tma_splitk_two_step_config(a, b, c, M, N, K):
    if N < 64 or N % 64 or K % 128:
        return None
    if not _tma_transposed_runtime_eligible(a, b, c, M, N, K):
        return None

    block_m = 16
    m_tiles = triton.cdiv(M, block_m)
    n_tiles = triton.cdiv(N, _TMA_TRANSPOSE_BLOCK_N)
    output_tiles = m_tiles * n_tiles
    deep_k = triton.cdiv(K, 128) > 32

    if output_tiles <= 1:
        block_k = 128
        split_k = 16
    elif deep_k and n_tiles >= 6 and output_tiles < 16:
        block_k = 64
        split_k = 16
    else:
        block_k = 128
        split_k = 8
    return block_m, block_k, split_k, 3, 4


def splitk_mm(a, b, c, M, N, K, op_name="mm"):
    logger.debug(
        "GEMS_NVIDIA MM_HOPPER, [op]: %s, [mm scenario]: splitk, [shape info]: [-, %s, %s, %s](batch, M, N, K)",
        op_name,
        M,
        N,
        K,
    )
    grid = lambda META: (
        triton.cdiv(M, max(META["BLOCK_M"], 16))
        * triton.cdiv(N, max(META["BLOCK_N"], 16)),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_kernel_splitk[grid](
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
            0,
            STORE_PARTIALS=False,
        )
    return c


def splitk_mm_two_step(a, b, c, M, N, K, op_name="mm"):
    split_k = _splitk_two_step_split_k(a, M, N, K)
    logger.debug(
        "GEMS_NVIDIA MM_HOPPER, [op]: %s, [mm scenario]: two-step splitk, "
        "[shape info]: [-, %s, %s, %s](batch, M, N, K), split_k=%s",
        op_name,
        M,
        N,
        K,
        split_k,
    )
    partials = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
    partial_grid = lambda META: (
        triton.cdiv(M, max(META["BLOCK_M"], 16))
        * triton.cdiv(N, max(META["BLOCK_N"], 16)),
        split_k,
    )
    n_elements = M * N
    reduce_block = 32 if n_elements <= 32 else 256
    reduce_warps = 1 if n_elements <= 32 else 4
    reduce_grid = (triton.cdiv(n_elements, reduce_block),)

    with torch_device_fn.device(a.device):
        mm_kernel_splitk_partials[partial_grid](
            a,
            b,
            partials,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            partials.stride(1),
            partials.stride(2),
            partials.stride(0),
            SPLIT_K=split_k,
            STORE_PARTIALS=True,
        )
        if c.is_contiguous():
            mm_kernel_tma_splitk_reduce[reduce_grid](
                partials,
                c,
                n_elements,
                partials.stride(0),
                SPLIT_K=split_k,
                BLOCK_SIZE=reduce_block,
                num_warps=reduce_warps,
            )
        else:
            mm_kernel_splitk_reduce_strided[reduce_grid](
                partials,
                c,
                M,
                N,
                c.stride(0),
                c.stride(1),
                partials.stride(0),
                SPLIT_K=split_k,
                BLOCK_SIZE=reduce_block,
                num_warps=reduce_warps,
            )
    return c


def streamk_scenario(a, b, M, N, K):
    # TODO: this my change sometime according to the realbenchmark result
    # Currently, the best configuration for streamk has only been tested on A100(capability[0] == 8).
    # The optimal settings for other devices need to be determined through real testing.
    eligible = (
        a.device == b.device
        and a.dtype in [torch.float16, torch.bfloat16]
        and b.dtype in [torch.float16, torch.bfloat16]
        and a.is_contiguous()
        and b.is_contiguous()
        and K > M * 5
        and K > N * 5
    )
    if not eligible:
        return False
    try:
        capability = torch.cuda.get_device_capability(a.device)
    except Exception:
        capability = get_device_capability()
    return capability[0] == 8


def splitk_scenario(a, b, M, N, K):
    if M <= 0 or N <= 0 or M >= 2048 or N >= 2048 or K < 4096:
        return False
    output_ctas = triton.cdiv(M, 16) * triton.cdiv(N, 32)
    sm_count = min(_get_sm_count_for_tensor(a), _H20_SM_COUNT)
    cta_limit = sm_count // 2
    if K >= 6144 and (
        (not b.is_contiguous() and b.T.is_contiguous())
        or (b.is_contiguous() and a.dtype == b.dtype == torch.bfloat16)
    ):
        cta_limit = 3 * sm_count // 4
    return output_ctas <= cta_limit


_WS_PLAN_SPLIT_M = 2
_WS_PLAN_FRAGMENTED_N = 3

_WS_TILES = {
    _WS_PLAN_SPLIT_M: (128, 128),
    _WS_PLAN_FRAGMENTED_N: (128, 256),
}


def _select_warp_specialized_plan(M, N, K):
    k_tiles = (K + 127) // 128
    aligned_k = not K % 128
    medium_k = aligned_k and 12 <= k_tiles <= 20
    deep_k = aligned_k and 32 <= k_tiles <= 64
    if medium_k or deep_k:
        n_ctas_128 = (N + 127) // 128
        if medium_k:
            if (65 <= M <= 80 or 97 <= M <= 104) and n_ctas_128 >= 8 * _H20_SM_COUNT:
                return _WS_PLAN_SPLIT_M
            if _H20_NEAR_WAVE_MIN_CTAS <= n_ctas_128 <= _H20_SM_COUNT:
                if 72 <= M <= 96:
                    return _WS_PLAN_SPLIT_M

        n_ctas_160 = (N + 159) // 160
        if medium_k:
            if (
                72 <= M <= 96
                and _H20_NEAR_WAVE_MIN_CTAS <= n_ctas_160 <= _H20_SM_COUNT
                and n_ctas_128 > _H20_SM_COUNT
            ):
                return _WS_PLAN_FRAGMENTED_N
        elif 65 <= M <= 104:
            if _H20_SM_COUNT < n_ctas_128 <= 2 * _H20_SM_COUNT:
                return _WS_PLAN_SPLIT_M
            if (
                M <= 72
                and _H20_SM_COUNT < n_ctas_160 <= 2 * _H20_SM_COUNT
                and n_ctas_128 > 2 * _H20_SM_COUNT
            ):
                return _WS_PLAN_FRAGMENTED_N
    return None


def _warp_specialized_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    nargs["a_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = (
        [block_k, block_n] if nargs["B_ROW_MAJOR"] else [block_n, block_k]
    )


def _estimate_warp_specialized_shared_memory_bytes(
    block_m, block_n, block_k, num_stages
):
    tile_bytes = (block_m * block_k + block_k * block_n) * 2
    return tile_bytes * num_stages + 8192


def _get_warp_specialized_mm_configs():
    configs = [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
            },
            num_stages=num_stages,
            num_warps=4,
            pre_hook=_warp_specialized_tma_set_block_size_hook,
        )
        for block_m, block_n in _WS_TILES.values()
        for block_k in (32, 64, 128)
        for num_stages in (2, 4, 8)
    ]
    shared_mem_limit = _get_shared_memory_limit_bytes()
    shared_mem_limit = min(shared_mem_limit or 227 * 1024, 220 * 1024)
    return [
        config
        for config in configs
        if _estimate_warp_specialized_shared_memory_bytes(
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.num_stages,
        )
        <= shared_mem_limit
    ]


def _prune_warp_specialized_configs(configs, named_args, **kwargs):
    M = kwargs.get("M", named_args.get("M"))
    N = kwargs.get("N", named_args.get("N"))
    K = kwargs.get("K", named_args.get("K"))
    plan = _select_warp_specialized_plan(M, N, K)
    fixed_tile = _WS_TILES.get(plan)
    if fixed_tile is None:
        return configs
    block_m, block_n = fixed_tile
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] == block_m and config.kwargs["BLOCK_N"] == block_n
    ]


if HAS_TLE_WARP_SPECIALIZATION:

    @tlc.builtin
    def _warp_specialized_smem_subslice(buf, offsets, shape, _semantic=None):
        offsets = [int(tlc._unwrap_if_constexpr(o)) for o in offsets]
        shape = [int(tlc._unwrap_if_constexpr(s)) for s in shape]
        result_ty = tle_types.buffered_tensor_type(
            buf.dtype,
            shape,
            buf.type.storage,
            buf.type.layout,
            _semantic,
            alloc_shape=buf.type.alloc_shape,
        )
        handle = _semantic.builder.create_memdesc_subslice(
            result_ty.to_ir(_semantic.builder), buf.handle, offsets
        )
        return tle_types.buffered_tensor(
            handle,
            buf.dtype,
            shape,
            buf.type.storage,
            buf.type.layout,
            _semantic,
            alloc_shape=buf.type.alloc_shape,
        )

    @triton.jit
    def _warp_specialized_mm_compute_split_partition(
        a_smem,
        b_smem,
        empty_a,
        empty_b,
        full_a,
        full_b,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        TILE_N: tl.constexpr,
        num_stages: tl.constexpr,
        CONSUMER_ID: tl.constexpr,
        WS_PLAN: tl.constexpr,
        B_ROW_MAJOR: tl.constexpr,
    ):
        fragmented: tl.constexpr = WS_PLAN == 3
        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, TILE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        tile_n_start = pid_n * TILE_N

        if CONSUMER_ID == 0:
            tl.static_assert(BLOCK_M == 128)
            if fragmented:
                tl.static_assert(BLOCK_N == 256 and TILE_N == 160)
            else:
                tl.static_assert(BLOCK_N == 128 and TILE_N == 128)
            acc_128 = tl.zeros((64, 128), dtype=tl.float32)
            if fragmented:
                acc_fragment = tl.zeros((64, 32), dtype=tl.float32)
            for k_block_idx in range(0, K // BLOCK_K):
                index = k_block_idx % num_stages
                phase = k_block_idx // num_stages
                tle_exp.gpu.barrier_wait(full_a[index], phaseIdx=phase)
                tle_exp.gpu.barrier_wait(full_b[index], phaseIdx=phase)
                a_head = _warp_specialized_smem_subslice(
                    a_smem.slot(index), [0, 0], [64, BLOCK_K]
                )
                b_full = b_smem.slot(index)
                if fragmented:
                    if B_ROW_MAJOR:
                        b_128 = _warp_specialized_smem_subslice(
                            b_full, [0, 0], [BLOCK_K, 128]
                        )
                        b_fragment = _warp_specialized_smem_subslice(
                            b_full, [0, 128], [BLOCK_K, 32]
                        )
                    else:
                        b_128 = _warp_specialized_smem_subslice(
                            b_full, [0, 0], [128, BLOCK_K]
                        )
                        b_fragment = _warp_specialized_smem_subslice(
                            b_full, [128, 0], [32, BLOCK_K]
                        )
                else:
                    b_128 = b_full
                acc_128 = tle_exp.gpu.wgmma(
                    a_head,
                    b_128,
                    acc_128,
                    out_dtype=tl.float32,
                    trans_b=not B_ROW_MAJOR,
                )
                acc_128 = tle_exp.gpu.wgmma_wait(0, acc_128)
                if fragmented:
                    acc_fragment = tle_exp.gpu.wgmma(
                        a_head,
                        b_fragment,
                        acc_fragment,
                        out_dtype=tl.float32,
                        trans_b=not B_ROW_MAJOR,
                    )
                    acc_fragment = tle_exp.gpu.wgmma_wait(0, acc_fragment)
                tle_exp.gpu.barrier_arrive(empty_a[index], phaseIdx=phase)
                tle_exp.gpu.barrier_arrive(empty_b[index], phaseIdx=phase)

            offs_m = pid_m * BLOCK_M + tl.arange(0, 64)
            n_128 = tile_n_start + tl.arange(0, 128)
            ptrs_128 = c_ptr + offs_m[:, None] * stride_cm + n_128[None, :] * stride_cn
            mask_128 = (offs_m[:, None] < M) & (n_128[None, :] < N)
            tl.store(ptrs_128, acc_128.to(c_ptr.dtype.element_ty), mask=mask_128)
            if fragmented:
                n_fragment = tile_n_start + 128 + tl.arange(0, 32)
                ptrs_fragment = (
                    c_ptr
                    + offs_m[:, None] * stride_cm
                    + n_fragment[None, :] * stride_cn
                )
                mask_fragment = (offs_m[:, None] < M) & (n_fragment[None, :] < N)
                tl.store(
                    ptrs_fragment,
                    acc_fragment.to(c_ptr.dtype.element_ty),
                    mask=mask_fragment,
                )
        else:
            tl.static_assert(BLOCK_M == 128)
            tl.static_assert(M > 64 and M <= BLOCK_M)
            split_tail: tl.constexpr = M > 96 and M <= 112
            if split_tail:
                tl.static_assert(not fragmented)
                if M <= 104:
                    final_tail_m: tl.constexpr = 8
                else:
                    final_tail_m: tl.constexpr = 16
                acc_tail_32_t = tl.zeros((128, 32), dtype=tl.float32)
                acc_final_tail_t = tl.zeros((128, final_tail_m), dtype=tl.float32)
                for k_block_idx in range(0, K // BLOCK_K):
                    index = k_block_idx % num_stages
                    phase = k_block_idx // num_stages
                    tle_exp.gpu.barrier_wait(full_a[index], phaseIdx=phase)
                    tle_exp.gpu.barrier_wait(full_b[index], phaseIdx=phase)
                    a_tail_32 = _warp_specialized_smem_subslice(
                        a_smem.slot(index), [64, 0], [32, BLOCK_K]
                    )
                    a_final_tail = _warp_specialized_smem_subslice(
                        a_smem.slot(index), [96, 0], [final_tail_m, BLOCK_K]
                    )
                    b_128 = b_smem.slot(index)
                    acc_tail_32_t = tle_exp.gpu.wgmma(
                        b_128,
                        a_tail_32,
                        acc_tail_32_t,
                        out_dtype=tl.float32,
                        trans_a=B_ROW_MAJOR,
                        trans_b=True,
                    )
                    acc_final_tail_t = tle_exp.gpu.wgmma(
                        b_128,
                        a_final_tail,
                        acc_final_tail_t,
                        out_dtype=tl.float32,
                        trans_a=B_ROW_MAJOR,
                        trans_b=True,
                    )
                    acc_tail_32_t = tle_exp.gpu.wgmma_wait(0, acc_tail_32_t)
                    acc_final_tail_t = tle_exp.gpu.wgmma_wait(0, acc_final_tail_t)
                    tle_exp.gpu.barrier_arrive(empty_a[index], phaseIdx=phase)
                    tle_exp.gpu.barrier_arrive(empty_b[index], phaseIdx=phase)

                n_128 = tile_n_start + tl.arange(0, 128)
                tail_32_rows = pid_m * BLOCK_M + 64 + tl.arange(0, 32)
                tail_32_ptrs = (
                    c_ptr
                    + tail_32_rows[:, None] * stride_cm
                    + n_128[None, :] * stride_cn
                )
                tail_32_mask = (tail_32_rows[:, None] < M) & (n_128[None, :] < N)
                tl.store(
                    tail_32_ptrs,
                    tl.trans(acc_tail_32_t).to(c_ptr.dtype.element_ty),
                    mask=tail_32_mask,
                )
                final_tail_rows = pid_m * BLOCK_M + 96 + tl.arange(0, final_tail_m)
                final_tail_ptrs = (
                    c_ptr
                    + final_tail_rows[:, None] * stride_cm
                    + n_128[None, :] * stride_cn
                )
                final_tail_mask = (final_tail_rows[:, None] < M) & (n_128[None, :] < N)
                tl.store(
                    final_tail_ptrs,
                    tl.trans(acc_final_tail_t).to(c_ptr.dtype.element_ty),
                    mask=final_tail_mask,
                )
            if M <= 72:
                tail_m: tl.constexpr = 8
            elif M <= 80:
                tail_m: tl.constexpr = 16
            elif M <= 96:
                tail_m: tl.constexpr = 32
            else:
                tail_m: tl.constexpr = 64
            acc_128_t = tl.zeros((128, tail_m), dtype=tl.float32)
            if fragmented:
                acc_64_t = tl.zeros((64, tail_m), dtype=tl.float32)
            for k_block_idx in range(0, 0 if split_tail else K // BLOCK_K):
                index = k_block_idx % num_stages
                phase = k_block_idx // num_stages
                tle_exp.gpu.barrier_wait(full_a[index], phaseIdx=phase)
                tle_exp.gpu.barrier_wait(full_b[index], phaseIdx=phase)
                a_tail = _warp_specialized_smem_subslice(
                    a_smem.slot(index), [64, 0], [tail_m, BLOCK_K]
                )
                b_full = b_smem.slot(index)
                if fragmented:
                    if B_ROW_MAJOR:
                        b_128 = _warp_specialized_smem_subslice(
                            b_full, [0, 0], [BLOCK_K, 128]
                        )
                        b_64 = _warp_specialized_smem_subslice(
                            b_full, [0, 128], [BLOCK_K, 64]
                        )
                    else:
                        b_128 = _warp_specialized_smem_subslice(
                            b_full, [0, 0], [128, BLOCK_K]
                        )
                        b_64 = _warp_specialized_smem_subslice(
                            b_full, [128, 0], [64, BLOCK_K]
                        )
                else:
                    b_128 = b_full
                acc_128_t = tle_exp.gpu.wgmma(
                    b_128,
                    a_tail,
                    acc_128_t,
                    out_dtype=tl.float32,
                    trans_a=B_ROW_MAJOR,
                    trans_b=True,
                )
                acc_128_t = tle_exp.gpu.wgmma_wait(0, acc_128_t)
                if fragmented:
                    acc_64_t = tle_exp.gpu.wgmma(
                        b_64,
                        a_tail,
                        acc_64_t,
                        out_dtype=tl.float32,
                        trans_a=B_ROW_MAJOR,
                        trans_b=True,
                    )
                    acc_64_t = tle_exp.gpu.wgmma_wait(0, acc_64_t)
                tle_exp.gpu.barrier_arrive(empty_a[index], phaseIdx=phase)
                tle_exp.gpu.barrier_arrive(empty_b[index], phaseIdx=phase)

            offs_m = pid_m * BLOCK_M + 64 + tl.arange(0, tail_m)
            n_128 = tile_n_start + tl.arange(0, 128)
            ptrs_128 = c_ptr + offs_m[:, None] * stride_cm + n_128[None, :] * stride_cn
            mask_128 = (offs_m[:, None] < M) & (n_128[None, :] < N) & (not split_tail)
            tl.store(
                ptrs_128,
                tl.trans(acc_128_t).to(c_ptr.dtype.element_ty),
                mask=mask_128,
            )
            if fragmented:
                n_64 = tile_n_start + 128 + tl.arange(0, 64)
                ptrs_64 = (
                    c_ptr + offs_m[:, None] * stride_cm + n_64[None, :] * stride_cn
                )
                mask_64 = (
                    (offs_m[:, None] < M)
                    & (n_64[None, :] < N)
                    & (n_64[None, :] < tile_n_start + TILE_N)
                )
                tl.store(
                    ptrs_64,
                    tl.trans(acc_64_t).to(c_ptr.dtype.element_ty),
                    mask=mask_64,
                )

    @triton.jit
    def _warp_specialized_mm_load_partition(
        a_desc,
        b_desc,
        a_smem,
        b_smem,
        empty_a,
        empty_b,
        full_a,
        full_b,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        TILE_N: tl.constexpr,
        num_stages: tl.constexpr,
        B_ROW_MAJOR: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, TILE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        m_start = pid_m * BLOCK_M
        n_start = pid_n * TILE_N

        for k_block_idx in range(0, K // BLOCK_K):
            index = k_block_idx % num_stages
            phase = k_block_idx // num_stages
            k_start = k_block_idx * BLOCK_K

            tle_exp.gpu.barrier_wait(empty_a[index], phaseIdx=phase)
            tle_exp.gpu.copy(
                a_desc,
                a_smem.slot(index),
                [BLOCK_M, BLOCK_K],
                [m_start, k_start],
                barrier=full_a[index],
            )
            tle_exp.gpu.barrier_wait(empty_b[index], phaseIdx=phase)
            if B_ROW_MAJOR:
                tle_exp.gpu.copy(
                    b_desc,
                    b_smem.slot(index),
                    [BLOCK_K, BLOCK_N],
                    [k_start, n_start],
                    barrier=full_b[index],
                )
            else:
                tle_exp.gpu.copy(
                    b_desc,
                    b_smem.slot(index),
                    [BLOCK_N, BLOCK_K],
                    [n_start, k_start],
                    barrier=full_b[index],
                )

    @libentry()
    @libtuner(
        configs=_get_warp_specialized_mm_configs(),
        key=["M", "N", "K", "stride_bk"],
        prune_configs_by={"early_config_prune": _prune_warp_specialized_configs},
        strategy=["default", "default", "default", "default"],
        policy="flagtune",
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_warp_specialized_tma",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        flagtune_pre_hook=_warp_specialized_tma_set_block_size_hook,
    )
    @triton.jit
    def mm_kernel_warp_specialized_tma(
        a_desc,
        b_desc,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_bk,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        num_warps: tl.constexpr,
        num_stages: tl.constexpr,
        WS_PLAN: tl.constexpr,
        B_ROW_MAJOR: tl.constexpr,
    ):
        _ = num_warps
        split_m: tl.constexpr = WS_PLAN == 2
        fragmented: tl.constexpr = WS_PLAN == 3
        tl.static_assert(split_m or fragmented)

        if split_m:
            tl.static_assert(BLOCK_M == 128 and BLOCK_N == 128)
        else:
            tl.static_assert(BLOCK_M == 128 and BLOCK_N == 256)

        a_smem = tle_exp.gpu.alloc(
            [num_stages, BLOCK_M, BLOCK_K],
            dtype=a_desc.dtype,
            layout=None,
            scope=tle_exp.gpu.smem,
        )
        if B_ROW_MAJOR:
            b_smem = tle_exp.gpu.alloc(
                [num_stages, BLOCK_K, BLOCK_N],
                dtype=b_desc.dtype,
                layout=None,
                scope=tle_exp.gpu.smem,
            )
        else:
            b_smem = tle_exp.gpu.alloc(
                [num_stages, BLOCK_N, BLOCK_K],
                dtype=b_desc.dtype,
                layout=None,
                scope=tle_exp.gpu.smem,
            )
        empty_a = tle_exp.gpu.alloc_barriers(
            num_barriers=num_stages,
            arrive_count=2,
            init=tle_exp.gpu.READY,
        )
        empty_b = tle_exp.gpu.alloc_barriers(
            num_barriers=num_stages,
            arrive_count=2,
            init=tle_exp.gpu.READY,
        )
        full_a = tle_exp.gpu.alloc_barriers(
            num_barriers=num_stages,
            arrive_count=1,
            expect_bytes=BLOCK_M * BLOCK_K * 2,
        )
        full_b = tle_exp.gpu.alloc_barriers(
            num_barriers=num_stages,
            arrive_count=1,
            expect_bytes=BLOCK_K * BLOCK_N * 2,
        )

        tile_n: tl.constexpr = 160 if fragmented else BLOCK_N
        if split_m:
            consumer_regs_0: tl.constexpr = 112
            consumer_regs_1: tl.constexpr = 112
        else:
            consumer_regs_0: tl.constexpr = 136
            consumer_regs_1: tl.constexpr = 112
        tle_exp.gpu.warp_specialize(
            [
                (
                    _warp_specialized_mm_load_partition,
                    (
                        a_desc,
                        b_desc,
                        a_smem,
                        b_smem,
                        empty_a,
                        empty_b,
                        full_a,
                        full_b,
                        M,
                        N,
                        K,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        tile_n,
                        num_stages,
                        B_ROW_MAJOR,
                    ),
                ),
                (
                    _warp_specialized_mm_compute_split_partition,
                    (
                        a_smem,
                        b_smem,
                        empty_a,
                        empty_b,
                        full_a,
                        full_b,
                        c_ptr,
                        M,
                        N,
                        K,
                        stride_cm,
                        stride_cn,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        tile_n,
                        num_stages,
                        0,
                        WS_PLAN,
                        B_ROW_MAJOR,
                    ),
                ),
                (
                    _warp_specialized_mm_compute_split_partition,
                    (
                        a_smem,
                        b_smem,
                        empty_a,
                        empty_b,
                        full_a,
                        full_b,
                        c_ptr,
                        M,
                        N,
                        K,
                        stride_cm,
                        stride_cn,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        tile_n,
                        num_stages,
                        1,
                        WS_PLAN,
                        B_ROW_MAJOR,
                    ),
                ),
            ],
            [4, 4],
            [consumer_regs_0, consumer_regs_1],
        )


def _warp_specialized_mm_runtime_eligible(a, b, c, M, N, K):
    descriptor_type = getattr(
        getattr(getattr(triton, "tools", None), "tensor_descriptor", None),
        "TensorDescriptor",
        None,
    )
    eligible = (
        HAS_TLE_WARP_SPECIALIZATION
        and descriptor_type is not None
        and hasattr(descriptor_type, "from_tensor")
        and a.is_cuda
        and b.is_cuda
        and c.is_cuda
        and a.device == b.device == c.device
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and c.dtype == torch.bfloat16
        and tuple(c.shape) == (M, N)
        and a.is_contiguous()
        and (b.is_contiguous() or b.T.is_contiguous())
        and c.is_contiguous()
        and _is_tma_descriptor_aligned(a)
        and _is_tma_descriptor_aligned(b, allow_transpose=True)
        and _is_tma_descriptor_aligned(c)
        and N % 32 == 0
        and K % 128 == 0
    )
    if not eligible:
        return False
    try:
        properties = torch.cuda.get_device_properties(a.device)
    except Exception:
        return False
    return properties.major == 9 and properties.multi_processor_count == _H20_SM_COUNT


def _select_warp_specialized_dispatch_plan(a, b, c, M, N, K):
    plan = _select_warp_specialized_plan(M, N, K)
    if plan is None:
        return None
    if not _warp_specialized_mm_runtime_eligible(a, b, c, M, N, K):
        return None
    return plan


def warp_specialized_mm_scenario(a, b, c, M, N, K):
    return _select_warp_specialized_dispatch_plan(a, b, c, M, N, K) is not None


def _warp_specialized_mm_descriptors(a, b, b_row_major):
    from triton.tools.tensor_descriptor import TensorDescriptor

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device=a.device)

    triton.set_allocator(alloc_fn)
    dummy_block = [1, 1]
    return (
        TensorDescriptor.from_tensor(a, block_shape=dummy_block),
        TensorDescriptor.from_tensor(
            b if b_row_major else b.T, block_shape=dummy_block
        ),
    )


def warp_specialized_mm(a, b, c, M, N, K):
    plan = _select_warp_specialized_plan(M, N, K)
    if plan is None:
        raise RuntimeError(f"no warp-specialized plan for shape {(M, N, K)}")
    logger.debug(
        "GEMS_NVIDIA MM_HOPPER, [mm scenario]: %s, "
        "[shape info]: [-, %s, %s, %s](batch, M, N, K)",
        "warp-specialized TMA plan %s" % plan,
        M,
        N,
        K,
    )
    b_row_major = b.stride(1) == 1
    a_desc, b_desc = _warp_specialized_mm_descriptors(a, b, b_row_major)
    if plan == _WS_PLAN_FRAGMENTED_N:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, 160),)
    else:
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
    with torch_device_fn.device(a.device):
        mm_kernel_warp_specialized_tma[grid](
            a_desc,
            b_desc,
            c,
            M=M,
            N=N,
            K=K,
            stride_bk=b.stride(0),
            stride_cm=c.stride(0),
            stride_cn=c.stride(1),
            WS_PLAN=plan,
            B_ROW_MAJOR=b_row_major,
        )
    return c


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
        "GEMS_NVIDIA M=%s N=%s K=%s a_col_major=%s b_col_major=%s",
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
    if M == 0 or N == 0:
        return c

    # Optimize for N=1 case (matrix-vector multiplication)
    if N == 1:
        return gemv_mm(a, b, c, M, K)
    if _splitk_gemv_scenario(a, b, c, M, N, K) or _batched_splitk_gemv_scenario(
        a, b, c, M, N, K
    ):
        return splitk_gemv_mm(a, b, c, M, N, K)
    # l2_cache_size = get_l2_cache_size()
    ws_plan = _select_warp_specialized_dispatch_plan(a, b, c, M, N, K)
    if ws_plan is not None:
        return warp_specialized_mm(a, b, c, M, N, K)
    if tma_transposed_direct_tuned_scenario(a, b, c, M, N, K):
        return tma_transposed_direct_tuned_mm(a, b, c, M, N, K)
    transposed_config = tma_transposed_config(a, b, c, M, N, K)
    if transposed_config is not None:
        return tma_transposed_mm(a, b, c, M, N, K, transposed_config)
    if streamk_scenario(a, b, M, N, K):
        sm_count = _get_sm_count_for_tensor(a)
        return streamk_mm(a, b, c, M, N, K, sm_count=sm_count)
    # if HAS_TLE and BLOCK_CLUSTER_MESH is not None:
    #     if cluster_remote_mm_scenario(a, b, c, M, N, K):
    #         return cluster_remote_mm(a, b, c, M, N, K)
    # Use splitk for small M
    if splitk_scenario(a, b, M, N, K):
        tma_splitk_config = _tma_splitk_two_step_config(a, b, c, M, N, K)
        if tma_splitk_config is not None:
            return tma_transposed_mm(
                a, b, c, M, N, K, tma_splitk_config, reduce_block=512
            )
        if (
            c.dtype == torch.float32
            and not torch.are_deterministic_algorithms_enabled()
        ):
            c.zero_()
            return splitk_mm(a, b, c, M, N, K)
        if c.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return splitk_mm_two_step(a, b, c, M, N, K)
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
    if M == 0 or N == 0:
        return out

    # Optimize for N=1 case (matrix-vector multiplication)
    if N == 1:
        return gemv_mm(a, b, out, M, K)
    if _splitk_gemv_scenario(a, b, out, M, N, K) or _batched_splitk_gemv_scenario(
        a, b, out, M, N, K
    ):
        return splitk_gemv_mm(a, b, out, M, N, K)
    # l2_cache_size = get_l2_cache_size()
    ws_plan = _select_warp_specialized_dispatch_plan(a, b, out, M, N, K)
    if ws_plan is not None:
        return warp_specialized_mm(a, b, out, M, N, K)
    if tma_transposed_direct_tuned_scenario(a, b, out, M, N, K):
        return tma_transposed_direct_tuned_mm(a, b, out, M, N, K)
    transposed_config = tma_transposed_config(a, b, out, M, N, K)
    if transposed_config is not None:
        return tma_transposed_mm(a, b, out, M, N, K, transposed_config)
    if streamk_scenario(a, b, M, N, K):
        sm_count = _get_sm_count_for_tensor(a)
        return streamk_mm(a, b, out, M, N, K, sm_count=sm_count)
    # if HAS_TLE and BLOCK_CLUSTER_MESH is not None:
    #     if cluster_remote_mm_scenario(a, b, out, M, N, K):
    #         return cluster_remote_mm(a, b, out, M, N, K)
    # Use splitk for small M
    if splitk_scenario(a, b, M, N, K):
        tma_splitk_config = _tma_splitk_two_step_config(a, b, out, M, N, K)
        if tma_splitk_config is not None:
            return tma_transposed_mm(
                a, b, out, M, N, K, tma_splitk_config, reduce_block=512
            )
        if (
            out.dtype == torch.float32
            and not torch.are_deterministic_algorithms_enabled()
        ):
            out.zero_()
            return splitk_mm(a, b, out, M, N, K)
        if out.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return splitk_mm_two_step(a, b, out, M, N, K)
    return general_mm(a, b, out, M, N, K)


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM for MoE router gate. weight shape: (N, K)."""
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    b = weight.t()
    if splitk_scenario(x, b, M, N, K):
        if not torch.are_deterministic_algorithms_enabled():
            c.zero_()
            return splitk_mm(x, b, c, M, N, K, op_name="router_gemm")
        return splitk_mm_two_step(x, b, c, M, N, K, op_name="router_gemm")
    return general_mm(x, b, c, M, N, K, op_name="router_gemm")
