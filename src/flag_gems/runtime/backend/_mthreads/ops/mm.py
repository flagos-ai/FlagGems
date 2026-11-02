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
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_mthreads_expand.yaml")
)

# Module-level capability flag: evaluated once at import time, then reused as
# a constant for the entire process lifetime with no repeated parsing overhead.
# False when Triton < 3.2 (e.g. 3.1), True when Triton >= 3.2.
SQMMA_ON = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 2)

# The warp-specialized SQMMA path below is built on the MUSA TLE frontend,
# which only exists in MUSA-enabled Triton builds.  Probe it once at import
# time; when it is missing every mm falls back to the ordinary SQMMA/FMA path.
try:
    import triton.experimental.tle.language as tle

    TLE_WS_ON = SQMMA_ON
except ImportError:  # pragma: no cover - depends on the installed Triton build
    tle = None
    TLE_WS_ON = False


def is_supported_sqmma_layout(tensor):
    return tensor.is_contiguous() or (
        tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
    )


def is_sqmma_compatible(a, b, N, K):
    return (
        SQMMA_ON
        and a.dim() == 2
        and b.dim() == 2
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and N % 8 == 0
        and K % 8 == 0
    )


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def mm_kernel(
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
    dtype: tl.constexpr,
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
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :]
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn), mask=mask_k[:, None]
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


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}),
    ],
    key=["M", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mm",
    flagtune_expand_op_name="gemv",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
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
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = ext.program_id(0)

    row_start = pid * BLOCK_M
    row_offset = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offset < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offset = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        a_ptrs = A + row_offset[:, None] * stride_am + k_offset[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        b_ptrs = B + k_offset * stride_bk
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        # Keep the reduction in fp32 so N=1 GEMV matches the mm path more closely.
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.sum(a * b[None, :], axis=1)

    c_ptrs = C + row_offset * stride_cm
    acc = acc.to(C.dtype.element_ty)
    tl.store(c_ptrs, acc, mask=row_mask)


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


def mm_fma(a, b):
    logger.debug("GEMS_MTHREADS MM_FMA")
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
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
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
            dtype=str(a.dtype).split(".")[-1],
            GROUP_M=8,
            IS_FP64=a.dtype == torch.float64,
        )
    return c


def gemv_mm(a, b, c, M, K):
    logger.debug(
        "GEMS_MTHREADS MM_GEMV_, [shape info]: [%s, %s, 1](M, K, N)",
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
            c.stride(0),
        )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_MTHREADS MM_OUT")
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
    c = out
    if N == 1:
        return gemv_mm(a, b, c, M, K)
    # The warp-specialized kernel stores straight into a row-major buffer whose
    # dtype matches the operands, so the SQMMA tiles and the output agree.  Any
    # other output layout or dtype keeps the portable FMA kernel.
    if (
        is_tle_ws_compatible(a, b, M, N, K)
        and out.is_contiguous()
        and out.dtype == a.dtype
        and out.shape == (M, N)
    ):
        with torch_device_fn.device(a.device):
            return mm_sqmma_ws(a, b, M, N, K, C=out)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
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
            dtype=str(a.dtype).split(".")[-1],
            GROUP_M=8,
            IS_FP64=a.dtype == torch.float64,
        )
    return c


def sqmma_descriptor_pre_hook(nargs):
    nargs["a_desc"].block_shape = [nargs["BLOCK_M"], nargs["BLOCK_K"]]
    nargs["b_desc"].block_shape = [nargs["BLOCK_K"], nargs["BLOCK_N"]]
    nargs["c_desc"].block_shape = [nargs["BLOCK_M"], nargs["BLOCK_N"]]


def _prune_sqmma_configs(configs, named_args, **kwargs):
    """Use the wider N tile only for measured large GEMM shapes."""
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    large = M >= 2048 and N >= 2048 and K >= 2048
    preferred_block_n = 256 if large else 128
    pruned = [
        config for config in configs if config.kwargs["BLOCK_N"] == preferred_block_n
    ]
    return pruned or list(configs)


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=1,
            num_warps=4,
            pre_hook=sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=1,
            num_warps=4,
            pre_hook=sqmma_descriptor_pre_hook,
        ),
    ],
    key=["M", "N", "K", "dtype"],
    strategy=["align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
    prune_configs_by={"early_config_prune": _prune_sqmma_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_sqmma",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=sqmma_descriptor_pre_hook,
)
@triton.jit
def mm_sqmma_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    dtype: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    offs_am = (pid_m * BLOCK_M).to(tl.int32)
    offs_bn = (pid_n * BLOCK_N).to(tl.int32)
    offs_k = 0
    offs_k = offs_k.to(tl.int32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_K
    tl.store_tensor_descriptor(c_desc, [offs_am, offs_bn], accumulator.to(c_desc.dtype))


def mm_sqmma(A, B, M, N, K):
    logger.debug("GEMS_MTHREADS MM_SQMMA")
    device = A.device
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    a_type = A.dtype
    b_type = B.dtype
    assert a_type == b_type, "Mat A and Mat B should have the same dtype"
    c_dtype = get_higher_dtype(a_type, b_type)
    C = torch.empty((M, N), dtype=c_dtype, device=device)
    desc_a = TensorDescriptor.from_tensor(A, [1, 1])
    desc_b = TensorDescriptor.from_tensor(B, [1, 1])
    desc_c = TensorDescriptor.from_tensor(C, [1, 1])
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        1,
        1,
    )
    mm_sqmma_kernel[grid](
        desc_a,
        desc_b,
        desc_c,
        M,
        N,
        K,
        str(a_type).split(".")[-1],
    )
    return C


@libentry()
@triton.jit
def mm_splitk_kernel(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    rm = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    pid_k = tl.program_id(2)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        ks = ((start * SPLIT_K + pid_k) * BLOCK_K + rk).to(tl.int64)
        a = tl.load(
            A + rm[:, None] * stride_am + ks[None, :] * stride_ak,
            (rm[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + ks[:, None] * stride_bk + rn[None, :] * stride_bn,
            (ks[:, None] < K) & (rn[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        P + pid_k * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def mm_splitk_reduce_kernel(
    P, C, SIZE: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, SPLIT_K)
    values = tl.load(
        P + parts[:, None] * SIZE + offsets[None, :],
        offsets[None, :] < SIZE,
        other=0,
    )
    tl.store(C + offsets, tl.sum(values, axis=0), offsets < SIZE)


def mm_small(a, b):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # Populate the S5000 MPs even for small output grids. Keep partials in
    # FP32 and reduce once, avoiding low-precision atomics and an output memset.
    split_k = 32
    partial = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
    with torch_device_fn.device(a.device):
        mm_splitk_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32), split_k)](
            a,
            b,
            partial,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=128,
            SPLIT_K=split_k,
            num_warps=4,
            num_stages=1,
        )
        mm_splitk_reduce_kernel[(triton.cdiv(M * N, 256),)](
            partial, c, M * N, split_k, BLOCK=256, num_warps=4
        )
    return c


# ---------------------------------------------------------------------------
# Warp-specialized SQMMA.  A producer warp group drives TME copies into a
# pipelined shared-memory ring while a consumer warp group runs SQMMA, so the
# next K tile is already in flight during the MMA instead of being serialized
# behind it.  This is what closes most of the remaining gap to the vendor
# persistent GEMM: the plain SQMMA kernel has no such overlap.
# ---------------------------------------------------------------------------

_WS_BLOCK_M = 256
_WS_BLOCK_N = 256
_WS_BLOCK_K = 64
_WS_PANEL_WIDTH = 4
_WS_NUM_WARPS = 16
_WS_STAGES = 3


@triton.jit
def _ws_rasterization_2d_column(
    block_idx,
    grid_x,
    grid_y,
    panel_width: tl.constexpr,
    full_panel: tl.constexpr,
):
    panel_size = panel_width * grid_y
    panel_idx = block_idx // panel_size
    panel_offset = block_idx % panel_size
    if full_panel:
        width = panel_width
    else:
        residual_panel_width = grid_x % panel_width
        full_panels_size = grid_x // panel_width * panel_width * grid_y
        width = tl.where(
            block_idx >= full_panels_size,
            residual_panel_width,
            panel_width,
        )
    row_idx = panel_offset // width
    mini_x = panel_offset % width
    mini_x = tl.where(row_idx % 2 == 1, width - 1 - mini_x, mini_x)
    row_idx = tl.where(panel_idx % 2 == 1, grid_y - 1 - row_idx, row_idx)
    col_idx = panel_idx * panel_width + mini_x
    return col_idx, row_idx


@triton.jit
def _ws_mm_consumer(
    a_reader,
    b_reader,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    offset_m = pid_m * block_m + tl.arange(0, block_m)
    offset_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in tl.range(
        0,
        k_tiles,
        num_stages=1,
        loop_unroll_factor=k_tiles,
    ):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    c_block_ptr = c_ptr + N * offset_m[:, None] + offset_n[None, :]
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ws_mm_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    K: tl.constexpr,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    offset_m = pid_m * block_m
    offset_n = pid_n * block_n
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    for k_iter in tl.range(
        0,
        k_tiles,
        num_stages=1,
        loop_unroll_factor=k_tiles,
    ):
        offset_k = k_iter * block_k
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(
            a_desc,
            a_slot.a,
            (block_m, block_k),
            (offset_m, offset_k),
        )
        tle.gpu.copy(
            b_desc,
            b_slot.b,
            (block_k, block_n),
            (offset_k, offset_n),
        )
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def mm_sqmma_ws_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    panel_width: tl.constexpr,
    pipeline_stages: tl.constexpr,
    full_panel: tl.constexpr,
):
    raw_bx = tl.program_id(axis=0)
    raw_by = tl.program_id(axis=1)
    grid_x = tl.num_programs(axis=0)
    grid_y = tl.num_programs(axis=1)
    block_idx = raw_bx + raw_by * grid_x
    pid_n, pid_m = _ws_rasterization_2d_column(
        block_idx,
        grid_x,
        grid_y,
        panel_width,
        full_panel,
    )

    a_smem = tle.gpu.alloc(
        (pipeline_stages, block_m, block_k),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (pipeline_stages, block_k, block_n),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="mm_a", a=a_smem)
    b_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="mm_b", b=b_smem)

    tle.gpu.warp_specialize(
        [
            (
                _ws_mm_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    M,
                    N,
                    K,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                ),
            ),
            (
                _ws_mm_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    K,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def is_tle_ws_compatible(a, b, M, N, K):
    """Warp-specialized SQMMA needs the TLE frontend and half-tile shapes."""
    return (
        TLE_WS_ON
        and M >= _WS_BLOCK_M // 2
        and N >= _WS_BLOCK_N // 2
        and is_sqmma_compatible(a, b, N, K)
    )


def mm_sqmma_ws(A, B, M, N, K, C=None):
    logger.debug("GEMS_MTHREADS MM_SQMMA_WS")
    device = A.device
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    assert A.dtype == B.dtype, "Mat A and Mat B should have the same dtype"
    if C is None:
        C = torch.empty(
            (M, N),
            dtype=get_higher_dtype(A.dtype, B.dtype),
            device=device,
        )
    desc_a = TensorDescriptor.from_tensor(A, [_WS_BLOCK_M, _WS_BLOCK_K])
    desc_b = TensorDescriptor.from_tensor(B, [_WS_BLOCK_K, _WS_BLOCK_N])
    grid = (triton.cdiv(N, _WS_BLOCK_N), triton.cdiv(M, _WS_BLOCK_M), 1)
    mm_sqmma_ws_kernel[grid](
        desc_a,
        desc_b,
        C,
        M,
        N,
        K,
        _WS_BLOCK_M,
        _WS_BLOCK_N,
        _WS_BLOCK_K,
        _WS_PANEL_WIDTH,
        _WS_STAGES,
        grid[0] % _WS_PANEL_WIDTH == 0,
        num_warps=_WS_NUM_WARPS,
        num_stages=_WS_STAGES,
    )
    return C


def mm(a, b):
    a_dtype = a.dtype
    b_dtype = b.dtype
    M, K = a.shape
    _, N = b.shape
    if N == 1:
        c_dtype = get_higher_dtype(a_dtype, b_dtype)
        c = torch.empty((M, N), device=a.device, dtype=c_dtype)
        return gemv_mm(a, b, c, M, K)

    # Split-K helps small output grids. Bound the partial workspace and require
    # more K work as the grid grows to amortize the separate reduction.
    if (
        a_dtype == b_dtype
        and a_dtype in (torch.float16, torch.bfloat16)
        and M > 0
        and N > 1
        and 512 <= K <= 32768
        and triton.cdiv(M, 32) * triton.cdiv(N, 32) <= min(64, K // 16)
    ):
        return mm_small(a, b)

    if is_tle_ws_compatible(a, b, M, N, K):
        return mm_sqmma_ws(
            a,
            b,
            M,
            N,
            K,
        )
    if is_sqmma_compatible(a, b, N, K):
        return mm_sqmma(
            a,
            b,
            M,
            N,
            K,
        )
    else:
        return mm_fma(a, b)
