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
import triton.experimental.tle.language as tle
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
# Shared with the upstream addmm/bmm warp-specialized paths.
TLE_WS_ON = SQMMA_ON


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


@libentry()
@triton.jit
def gemv_long_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BATCH_M: tl.constexpr = 1,
    BATCH_STRIDE_B: tl.constexpr = 0,
    BATCH_STRIDE_C: tl.constexpr = 0,
):
    batch = tl.program_id(0) % BATCH_M
    rows = tl.program_id(0) // BATCH_M * BLOCK_M + tl.arange(0, BLOCK_M)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    # Defer the horizontal reduction until all K tiles have been accumulated.
    for start in range(tl.cdiv(K, BLOCK_K)):
        offsets = start * BLOCK_K + ks
        a = tl.load(
            A + rows[:, None] * stride_am + offsets[None, :] * stride_ak,
            (rows[:, None] < M) & (offsets[None, :] < K),
            other=0,
        ).to(tl.float32)
        if BATCH_M > 1:
            # Matching the two load layouts avoids broadcasting the input
            # vector across warps.
            batches = tl.broadcast_to(batch, (BLOCK_M,))
            b = tl.load(
                B + batches[:, None] * BATCH_STRIDE_B + offsets[None, :] * stride_bk,
                offsets[None, :] < K,
                other=0,
            ).to(tl.float32)
            acc = tl.fma(b, a, acc)
        else:
            b = tl.load(B + offsets * stride_bk, offsets < K, other=0).to(tl.float32)
            acc = tl.fma(a, b[None, :], acc)
    tl.store(
        C + batch * BATCH_STRIDE_C + rows * stride_cm,
        tl.sum(acc, axis=1),
        rows < M,
    )


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
    c = torch.empty((M, N), device=device, dtype=a.dtype)
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
    if K >= 65536:
        low_precision = a.dtype in (torch.float16, torch.bfloat16)
        block_m = 2 if low_precision else 4
        with torch_device_fn.device(a.device):
            gemv_long_kernel[(triton.cdiv(M, block_m),)](
                a,
                b,
                c,
                M,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                c.stride(0),
                BLOCK_M=block_m,
                BLOCK_K=2048 if low_precision else 1024,
                num_warps=4,
                num_stages=1,
            )
        return c

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
    C = torch.empty((M, N), dtype=a_type, device=device)
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
    parts = tl.arange(0, triton.next_power_of_2(SPLIT_K))
    values = tl.load(
        P + parts[:, None] * SIZE + offsets[None, :],
        (parts[:, None] < SPLIT_K) & (offsets[None, :] < SIZE),
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
            dtype=A.dtype,
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


def prune_mm_fp32_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if args["TRANSPOSE_OUTPUT"]:
        tiles = {(128, 32, 32, True), (64, 64, 32, True)}
    elif args["CONTIGUOUS_K"]:
        tiles = {(64, 128, 16, True)}
    elif args["M"] <= 8:
        tiles = {(16, 64, 32, False), (16, 128, 32, False)}
    else:
        tiles = {(64, 64, 32, False), (64, 128, 32, True)}
    return [
        config
        for config in configs
        if tuple(
            config.kwargs[key]
            for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "enable_backend_opt")
        )
        in tiles
    ]


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_K": bk,
                "enable_backend_opt": opt,
            },
            num_warps=4,
            num_stages=1,
        )
        for bm, bn, bk, opt in (
            (16, 64, 32, False),
            (16, 128, 32, False),
            (64, 64, 32, False),
            (64, 128, 32, True),
            (128, 32, 32, True),
            (64, 64, 32, True),
            (64, 128, 16, True),
        )
    ],
    key=[
        "M",
        "N",
        "K",
        "stride_am",
        "stride_ak",
        "stride_bk",
        "stride_bn",
        "SPLIT_K",
        "TRANSPOSE_OUTPUT",
        "CONTIGUOUS_K",
    ],
    prune_configs_by={"early_config_prune": prune_mm_fp32_configs},
)
@triton.heuristics({"BLOCK_M": lambda args: max(16, args["BLOCK_M"])})
@triton.jit
def mm_fp32_kernel(
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
    INDEX_DTYPE: tl.constexpr = tl.int64,
    TRANSPOSE_OUTPUT: tl.constexpr = False,
    CONTIGUOUS_K: tl.constexpr = False,
    REBASE_B: tl.constexpr = False,
):
    # Keep at least 16 rows so TF32x3 uses the matrix instruction path.
    rm = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(INDEX_DTYPE)
    rn = (tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)).to(INDEX_DTYPE)
    rk = tl.arange(0, BLOCK_K)
    if REBASE_B:
        B += (tl.program_id(1) * BLOCK_N).to(tl.int64) * stride_bn
        bn_offsets = tl.arange(0, BLOCK_N)
    else:
        bn_offsets = rn
    pid_k = tl.program_id(2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if CONTIGUOUS_K:
            k_block = pid_k * tl.cdiv(K, BLOCK_K * SPLIT_K) + start
        else:
            k_block = start * SPLIT_K + pid_k
        ks = (k_block * BLOCK_K + rk).to(INDEX_DTYPE)
        a = tl.load(
            A + rm[:, None] * stride_am + ks[None, :] * stride_ak,
            (rm[:, None] < M) & (ks[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + ks[:, None] * stride_bk + bn_offsets[None, :] * stride_bn,
            (ks[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        # Three TF32 products retain the FP32 residuals in the tensor-core path.
        acc = tl.dot(a, b, acc, input_precision="tf32x3")
    if TRANSPOSE_OUTPUT:
        offsets = rn[None, :] * M + rm[:, None]
    else:
        offsets = rm[:, None] * N + rn[None, :]
    tl.store(
        P + pid_k.to(INDEX_DTYPE) * M * N + offsets,
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def mm_fp32_transpose_kernel(
    B,
    P,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rk = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    rn = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    values = tl.load(
        B + rn[:, None] * stride_bn + rk[None, :] * stride_bk,
        (rn[:, None] < N) & (rk[None, :] < K),
        other=0.0,
    )
    tl.store(
        P + rk[:, None] * N + rn[None, :],
        tl.trans(values),
        (rk[:, None] < K) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def mm_vector_fp32_kernel(
    A,
    B,
    P,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    pid_k = tl.program_id(1)
    if TRANS_B:
        acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    else:
        acc = tl.zeros((BLOCK_K, BLOCK_N), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        ks = (start * SPLIT_K + pid_k) * BLOCK_K + rk
        a = tl.load(A + ks, ks < K, other=0.0)
        if TRANS_B:
            b = tl.load(
                B + rn[:, None] * K + ks[None, :],
                (rn[:, None] < N) & (ks[None, :] < K),
                other=0.0,
            )
            acc = tl.fma(a[None, :], b, acc)
        else:
            b = tl.load(
                B + ks[:, None] * N + rn[None, :],
                (ks[:, None] < K) & (rn[None, :] < N),
                other=0.0,
            )
            acc = tl.fma(a[:, None], b, acc)
    if TRANS_B:
        values = tl.sum(acc, axis=1)
    else:
        values = tl.sum(acc, axis=0)
    tl.store(P + pid_k * N + rn, values, rn < N)


def mm_fp32(a, b):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    trans_b = b.stride(0) == 1
    vector = M == 1 and a.is_contiguous() and is_supported_sqmma_layout(b)
    transpose_output = False
    contiguous_k = False
    index_dtype = tl.int64

    if vector:
        block_n, block_k, split_k = (4, 512, 1) if trans_b else (32, 128, 8)
    elif M <= 8:
        split_k = 8 if trans_b else 4
    elif M <= 256:
        split_k = (
            1 if not trans_b and M <= 128 and N >= 8192 and 4096 <= K <= 8192 else 4
        )
        transpose_output = trans_b and N <= 6144
        contiguous_k = trans_b and not transpose_output
    else:
        # Many output rows amortize one packing pass over a strided weight.
        if not b.is_contiguous():
            packed = torch.empty((K, N), device=b.device, dtype=b.dtype)
            with torch_device_fn.device(a.device):
                mm_fp32_transpose_kernel[(triton.cdiv(K, 64), triton.cdiv(N, 64))](
                    b, packed, K, N, b.stride(0), b.stride(1), 64, num_warps=4
                )
            b = packed
        split_k = 32 if K > 32768 else 1

    if not vector and M > 8 and (M <= 256 or K > 32768):
        a_span = (M - 1) * a.stride(0) + (K - 1) * a.stride(1) + 1
        b_span = (K - 1) * b.stride(0) + (N - 1) * b.stride(1) + 1
        if 4 * max(a_span, b_span, split_k * M * N) < 2**31:
            index_dtype = tl.int32
    partial = (
        torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
        if split_k > 1
        else c
    )
    if transpose_output:
        # With few rows, read the NT weight directly through B.T @ A.T.
        a, b = b.T, a.T
        M, N = N, M
    with torch_device_fn.device(a.device):
        if vector:
            mm_vector_fp32_kernel[(triton.cdiv(N, block_n), split_k)](
                a,
                b,
                partial,
                N,
                K,
                trans_b,
                block_n,
                block_k,
                split_k,
                num_warps=4,
                num_stages=1,
            )
        else:
            mm_fp32_kernel[
                lambda meta: (
                    triton.cdiv(M, meta["BLOCK_M"]),
                    triton.cdiv(N, meta["BLOCK_N"]),
                    split_k,
                )
            ](
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
                SPLIT_K=split_k,
                INDEX_DTYPE=index_dtype,
                TRANSPOSE_OUTPUT=transpose_output,
                CONTIGUOUS_K=contiguous_k,
                REBASE_B=contiguous_k,
            )
        if split_k > 1:
            mm_splitk_reduce_kernel[(triton.cdiv(M * N, 256),)](
                partial, c, M * N, split_k, BLOCK=256, num_warps=4
            )
    return c


@triton.jit
def mm_pipe_producer(
    a_desc,
    b_desc,
    writer,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    for i in range(triton.cdiv(K, BLOCK_K)):
        slot = writer.acquire(i)
        tle.gpu.copy(a_desc, slot.a, (BLOCK_M, BLOCK_K), offsets=(row, i * BLOCK_K))
        if TRANS_B:
            tle.gpu.copy(b_desc, slot.b, (BLOCK_N, BLOCK_K), offsets=(col, i * BLOCK_K))
        else:
            tle.gpu.copy(b_desc, slot.b, (BLOCK_K, BLOCK_N), offsets=(i * BLOCK_K, col))
        writer.commit(i)


@triton.jit
def mm_pipe_consumer(
    C,
    reader,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for i in range(triton.cdiv(K, BLOCK_K)):
        ready = reader.wait(i)
        acc = tle.gpu.wgmma(ready.slot.a, ready.slot.b, acc, trans_b=TRANS_B)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(i)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@libentry()
@triton.jit
def mm_pipe_kernel(
    a_desc,
    b_desc,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 128,
    CAPACITY: tl.constexpr = 4,
):
    a = tle.gpu.alloc((CAPACITY, BLOCK_M, BLOCK_K), a_desc.dtype)
    if TRANS_B:
        b = tle.gpu.alloc((CAPACITY, BLOCK_N, BLOCK_K), b_desc.dtype)
    else:
        b = tle.gpu.alloc((CAPACITY, BLOCK_K, BLOCK_N), b_desc.dtype)
    pipe = tle.pipe(capacity=CAPACITY, a=a, b=b)
    tle.gpu.warp_specialize(
        (
            (
                mm_pipe_consumer,
                (C, pipe.reader(), M, N, K, TRANS_B, BLOCK_N, BLOCK_M, BLOCK_K),
            ),
            (
                mm_pipe_producer,
                (a_desc, b_desc, pipe.writer(), K, TRANS_B, BLOCK_N, BLOCK_M, BLOCK_K),
            ),
        ),
        (4,),
        (64,),
    )


def gemv_nt_mm(a, b, N, K):
    batches = a.shape[0]
    c = torch.empty((batches, N), device=a.device, dtype=a.dtype)
    with torch_device_fn.device(a.device):
        gemv_long_kernel[(triton.cdiv(N, 2) * batches,)](
            b,
            a,
            c,
            N,
            K,
            b.stride(1),
            b.stride(0),
            a.stride(1),
            1,
            BLOCK_M=2,
            BLOCK_K=512 if batches > 1 else triton.next_power_of_2(K),
            BATCH_M=batches,
            BATCH_STRIDE_B=a.stride(0),
            BATCH_STRIDE_C=c.stride(0),
            num_warps=4 if batches > 1 else 8,
            num_stages=1,
            enable_backend_opt=True,
        )
    return c


@triton.jit
def mm_wide_tile(
    M: tl.constexpr, N: tl.constexpr, BLOCK_TAIL: tl.constexpr, GROUP_M: tl.constexpr
):
    if GROUP_M > 0:
        grid_m = tl.cdiv(M, 256)
        grid_n = tl.cdiv(N, 256 + BLOCK_TAIL)
        pid = tl.program_id(0)
        group_width = GROUP_M * grid_n
        group_id = pid // group_width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        within_group = pid % group_width
        return (
            group_id * GROUP_M + within_group % group_size,
            within_group // group_size,
        )
    else:
        return tl.program_id(0), tl.program_id(1)


@triton.jit
def mm_wide_tle_producer(
    a_desc,
    b_desc,
    b_tail_desc,
    writer,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = mm_wide_tile(M, N, BLOCK_TAIL, GROUP_M)
    offs_m = pid_m * 256
    offs_n = pid_n * (256 + BLOCK_TAIL)
    iterations = tl.cdiv(K, BLOCK_K * SPLIT_K)
    offs_k = 0
    if SPLIT_K > 1:
        offs_k = tl.program_id(2) * iterations * BLOCK_K
    for i in range(iterations):
        slot = writer.acquire(i)
        tle.gpu.copy(
            a_desc, slot.a, (256, BLOCK_K), offsets=(offs_m, offs_k + i * BLOCK_K)
        )
        if TRANS_B:
            tle.gpu.copy(
                b_desc, slot.b, (256, BLOCK_K), offsets=(offs_n, offs_k + i * BLOCK_K)
            )
            tle.gpu.copy(
                b_tail_desc,
                slot.b_tail,
                (BLOCK_TAIL, BLOCK_K),
                offsets=(offs_n + 256, offs_k + i * BLOCK_K),
            )
        else:
            tle.gpu.copy(
                b_desc, slot.b, (BLOCK_K, 256), offsets=(offs_k + i * BLOCK_K, offs_n)
            )
            tle.gpu.copy(
                b_tail_desc,
                slot.b_tail,
                (BLOCK_K, BLOCK_TAIL),
                offsets=(offs_k + i * BLOCK_K, offs_n + 256),
            )
        writer.commit(i)


@triton.jit
def mm_wide_tle_consumer(
    C,
    reader,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Two independent accumulators share A across adjacent column tiles.
    acc = tl.zeros((256, 256), dtype=tl.float32)
    acc_tail = tl.zeros((256, BLOCK_TAIL), dtype=tl.float32)
    for i in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        result = reader.wait(i)
        acc = tle.gpu.wgmma(result.slot.a, result.slot.b, acc, trans_b=TRANS_B)
        acc_tail = tle.gpu.wgmma(
            result.slot.a, result.slot.b_tail, acc_tail, trans_b=TRANS_B
        )
        acc = tle.gpu.wgmma_wait(0, acc)
        acc_tail = tle.gpu.wgmma_wait(0, acc_tail)
        reader.release(i)
    wide_offsets: tl.constexpr = (
        SPLIT_K * M * N * (C.dtype.element_ty.primitive_bitwidth // 8) >= 2**31
    )
    if SPLIT_K > 1:
        if wide_offsets:
            C += tl.program_id(2).to(tl.int64) * M * N
        else:
            C += tl.program_id(2) * M * N
    pid_m, pid_n = mm_wide_tile(M, N, BLOCK_TAIL, GROUP_M)
    rows = pid_m * 256 + tl.arange(0, 256)
    cols = pid_n * (256 + BLOCK_TAIL) + tl.arange(0, 256)
    cols_tail = pid_n * (256 + BLOCK_TAIL) + 256 + tl.arange(0, BLOCK_TAIL)
    row_offsets = rows[:, None].to(tl.int64) * N if wide_offsets else rows[:, None] * N
    tl.store(
        C + row_offsets + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(
        C + row_offsets + cols_tail[None, :],
        acc_tail,
        (rows[:, None] < M) & (cols_tail[None, :] < N),
    )


@libentry()
@triton.jit
def mm_wide_tle_kernel(
    a_desc,
    b_desc,
    b_tail_desc,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    DTYPE: tl.constexpr = None,
    BLOCK_TAIL: tl.constexpr = 64,
    GROUP_M: tl.constexpr = 0,
):
    a = tle.gpu.alloc((STAGES, 256, BLOCK_K), a_desc.dtype)
    if TRANS_B:
        b = tle.gpu.alloc((STAGES, 256, BLOCK_K), b_desc.dtype)
        b_tail = tle.gpu.alloc((STAGES, BLOCK_TAIL, BLOCK_K), b_tail_desc.dtype)
    else:
        b = tle.gpu.alloc((STAGES, BLOCK_K, 256), b_desc.dtype)
        b_tail = tle.gpu.alloc((STAGES, BLOCK_K, BLOCK_TAIL), b_tail_desc.dtype)
    pipe = tle.pipe(capacity=STAGES, a=a, b=b, b_tail=b_tail)
    tle.gpu.warp_specialize(
        (
            (
                mm_wide_tle_consumer,
                (
                    C,
                    pipe.reader(),
                    M,
                    N,
                    K,
                    TRANS_B,
                    BLOCK_K,
                    SPLIT_K,
                    BLOCK_TAIL,
                    GROUP_M,
                ),
            ),
            (
                mm_wide_tle_producer,
                (
                    a_desc,
                    b_desc,
                    b_tail_desc,
                    pipe.writer(),
                    M,
                    N,
                    K,
                    TRANS_B,
                    BLOCK_K,
                    SPLIT_K,
                    BLOCK_TAIL,
                    GROUP_M,
                ),
            ),
        ),
        (4,),
        (64,),
    )


@triton.jit
def mm_tall_tile(
    M: tl.constexpr, N: tl.constexpr, BLOCK_TAIL: tl.constexpr, GROUP_M: tl.constexpr
):
    grid_m = tl.cdiv(M, 256 + BLOCK_TAIL)
    grid_n = tl.cdiv(N, 256)
    pid = tl.program_id(0)
    group_id = pid // (GROUP_M * grid_n)
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    within_group = pid % (GROUP_M * grid_n)
    return group_id * GROUP_M + within_group % group_size, within_group // group_size


@triton.jit
def mm_tall_tle_producer(
    a_desc,
    a_tail_desc,
    b_desc,
    writer,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = mm_tall_tile(M, N, BLOCK_TAIL, GROUP_M)
    offs_m = pid_m * (256 + BLOCK_TAIL)
    offs_n = pid_n * 256
    for i in range(tl.cdiv(K, 32)):
        slot = writer.acquire(i)
        tle.gpu.copy(a_desc, slot.a, (256, 32), offsets=(offs_m, i * 32))
        tle.gpu.copy(
            a_tail_desc, slot.a_tail, (BLOCK_TAIL, 32), offsets=(offs_m + 256, i * 32)
        )
        if TRANS_B:
            tle.gpu.copy(b_desc, slot.b, (256, 32), offsets=(offs_n, i * 32))
        else:
            tle.gpu.copy(b_desc, slot.b, (32, 256), offsets=(i * 32, offs_n))
        writer.commit(i)


@triton.jit
def mm_tall_tle_consumer(
    C,
    reader,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Reuse B across adjacent row tiles with independent accumulators.
    acc = tl.zeros((256, 256), dtype=tl.float32)
    acc_tail = tl.zeros((BLOCK_TAIL, 256), dtype=tl.float32)
    for i in range(tl.cdiv(K, 32)):
        result = reader.wait(i)
        acc = tle.gpu.wgmma(result.slot.a, result.slot.b, acc, trans_b=TRANS_B)
        acc_tail = tle.gpu.wgmma(
            result.slot.a_tail, result.slot.b, acc_tail, trans_b=TRANS_B
        )
        acc = tle.gpu.wgmma_wait(0, acc)
        acc_tail = tle.gpu.wgmma_wait(0, acc_tail)
        reader.release(i)
    pid_m, pid_n = mm_tall_tile(M, N, BLOCK_TAIL, GROUP_M)
    rows = pid_m * (256 + BLOCK_TAIL) + tl.arange(0, 256)
    rows_tail = pid_m * (256 + BLOCK_TAIL) + 256 + tl.arange(0, BLOCK_TAIL)
    cols = pid_n * 256 + tl.arange(0, 256)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(
        C + rows_tail[:, None] * N + cols[None, :],
        acc_tail,
        (rows_tail[:, None] < M) & (cols[None, :] < N),
    )


@libentry()
@triton.jit
def mm_tall_tle_kernel(
    a_desc,
    a_tail_desc,
    b_desc,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_TAIL: tl.constexpr = 64,
    GROUP_M: tl.constexpr = 2,
):
    a = tle.gpu.alloc((4, 256, 32), a_desc.dtype)
    a_tail = tle.gpu.alloc((4, BLOCK_TAIL, 32), a_tail_desc.dtype)
    if TRANS_B:
        b = tle.gpu.alloc((4, 256, 32), b_desc.dtype)
    else:
        b = tle.gpu.alloc((4, 32, 256), b_desc.dtype)
    pipe = tle.pipe(capacity=4, a=a, a_tail=a_tail, b=b)
    tle.gpu.warp_specialize(
        (
            (
                mm_tall_tle_consumer,
                (C, pipe.reader(), M, N, K, TRANS_B, BLOCK_TAIL, GROUP_M),
            ),
            (
                mm_tall_tle_producer,
                (
                    a_desc,
                    a_tail_desc,
                    b_desc,
                    pipe.writer(),
                    M,
                    N,
                    K,
                    TRANS_B,
                    BLOCK_TAIL,
                    GROUP_M,
                ),
            ),
        ),
        (4,),
        (64,),
    )


@triton.jit
def mm_stream_tme_producer(
    A,
    B,
    writer,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TRANS_B: tl.constexpr,
    CONSUMER_WARPS: tl.constexpr,
    INNER_CACHE: tl.constexpr,
    OUTER_CACHE: tl.constexpr,
    CACHE_A: tl.constexpr,
    LEADER_ONLY: tl.constexpr,
):
    im = tl.program_id(0) * BLOCK_M
    jn = tl.program_id(1) * BLOCK_N
    count = tl.cdiv(K, BLOCK_K * SPLIT_K)
    startk = tl.program_id(2) * count * BLOCK_K
    # Keep the last transfer outside the multi-thread producer loop. This
    # avoids LLVM jump threading duplicating its leader/nonleader latch.
    for part in tl.static_range(1 if LEADER_ONLY else 2):
        begin = 0 if part == 0 else count - 1
        end = count if LEADER_ONLY or part == 1 else count - 1
        for i in range(begin, end):
            slot = writer.acquire(i)
            policy: tl.constexpr = ("none", "once", "normal", "persist")
            tle.gpu.copy(
                A,
                slot.a,
                (BLOCK_M, BLOCK_K),
                (im, startk + i * BLOCK_K),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner=policy[2 if CACHE_A else INNER_CACHE],
                    outer=policy[2 if CACHE_A else OUTER_CACHE],
                ),
            )
            if TRANS_B:
                tle.gpu.copy(
                    B,
                    slot.b,
                    (BLOCK_N, BLOCK_K),
                    (jn, startk + i * BLOCK_K),
                    cache_policy=tle.gpu.mthreads.TMECachePolicy(
                        inner=policy[INNER_CACHE], outer=policy[OUTER_CACHE]
                    ),
                )
            else:
                tle.gpu.copy(
                    B,
                    slot.b,
                    (BLOCK_K, BLOCK_N),
                    (startk + i * BLOCK_K, jn),
                    cache_policy=tle.gpu.mthreads.TMECachePolicy(
                        inner=policy[INNER_CACHE], outer=policy[OUTER_CACHE]
                    ),
                )
            writer.commit(i)


@triton.jit
def mm_stream_tme_consumer(
    C,
    reader,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for i in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        result = reader.wait(i)
        acc = tle.gpu.wgmma(result.slot.a, result.slot.b, acc, trans_b=TRANS_B)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(i)
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        C + tl.program_id(2) * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@libentry()
@triton.jit
def mm_stream_tme_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TRANS_B: tl.constexpr,
    CONSUMER_WARPS: tl.constexpr,
    INNER_CACHE: tl.constexpr,
    OUTER_CACHE: tl.constexpr,
    DTYPE: tl.constexpr,
    CACHE_A: tl.constexpr = False,
    LEADER_ONLY: tl.constexpr = False,
):
    aa = tle.gpu.alloc((STAGES, BLOCK_M, BLOCK_K), A.dtype)
    if TRANS_B:
        bb = tle.gpu.alloc((STAGES, BLOCK_N, BLOCK_K), B.dtype)
    else:
        bb = tle.gpu.alloc((STAGES, BLOCK_K, BLOCK_N), B.dtype)
    pipe = tle.pipe(capacity=STAGES, a=aa, b=bb)
    tle.gpu.warp_specialize(
        (
            (
                mm_stream_tme_consumer,
                (
                    C,
                    pipe.reader(),
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    STAGES,
                    SPLIT_K,
                    TRANS_B,
                ),
            ),
            (
                mm_stream_tme_producer,
                (
                    A,
                    B,
                    pipe.writer(),
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    STAGES,
                    SPLIT_K,
                    TRANS_B,
                    CONSUMER_WARPS,
                    INNER_CACHE,
                    OUTER_CACHE,
                    CACHE_A,
                    LEADER_ONLY,
                ),
            ),
        ),
        (4,),
        (64,),
    )


@triton.jit
def mm_ilp_stream_producer(
    A,
    B,
    B_tail,
    writer,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_TAIL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    TRANS_B: tl.constexpr,
    CONSUMER_WARPS: tl.constexpr,
    CACHE_A: tl.constexpr,
    SPLIT_K: tl.constexpr,
    LEADER_ONLY: tl.constexpr,
    ROTATE_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M
    offs_n = tl.program_id(1) * (BLOCK_N + BLOCK_N_TAIL)
    count = 2 * tl.cdiv(K, 2 * BLOCK_K * SPLIT_K)
    offs_k = tl.program_id(2) * count * BLOCK_K if SPLIT_K > 1 else 0
    for i in range(count):
        k_tile = (i + tl.program_id(1) * ROTATE_K) % count if ROTATE_K else i
        slot = writer.acquire(i)
        tle.gpu.copy(
            A,
            slot.a,
            (BLOCK_M, BLOCK_K),
            (offs_m, offs_k + k_tile * BLOCK_K),
            cache_policy=(
                tle.gpu.mthreads.TMECachePolicy(inner="normal", outer="normal")
                if CACHE_A
                else tle.gpu.mthreads.TMECachePolicy(inner="once", outer="none")
            ),
        )
        if TRANS_B:
            tle.gpu.copy(
                B,
                slot.b,
                (BLOCK_N, BLOCK_K),
                (offs_n, offs_k + k_tile * BLOCK_K),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="once", outer="none"
                ),
            )
            tle.gpu.copy(
                B_tail,
                slot.b_tail,
                (BLOCK_N_TAIL, BLOCK_K),
                (offs_n + BLOCK_N, offs_k + k_tile * BLOCK_K),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="once", outer="none"
                ),
            )
        else:
            tle.gpu.copy(
                B,
                slot.b,
                (BLOCK_K, BLOCK_N),
                (offs_k + k_tile * BLOCK_K, offs_n),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="once", outer="none"
                ),
            )
            tle.gpu.copy(
                B_tail,
                slot.b_tail,
                (BLOCK_K, BLOCK_N_TAIL),
                (offs_k + k_tile * BLOCK_K, offs_n + BLOCK_N),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="once", outer="none"
                ),
            )
        writer.commit(i)


@triton.jit
def mm_ilp_stream_consumer(
    C,
    reader,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_TAIL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    TRANS_B: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Independent even/odd K accumulators overlap four SQMMA instructions.
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_tail = tl.zeros((BLOCK_M, BLOCK_N_TAIL), tl.float32)
    acc_odd = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_tail_odd = tl.zeros((BLOCK_M, BLOCK_N_TAIL), tl.float32)
    for i in range(tl.cdiv(K, 2 * BLOCK_K * SPLIT_K)):
        even = reader.wait(2 * i)
        acc = tle.gpu.wgmma(even.slot.a, even.slot.b, acc, trans_b=TRANS_B)
        acc_tail = tle.gpu.wgmma(
            even.slot.a, even.slot.b_tail, acc_tail, trans_b=TRANS_B
        )
        odd = reader.wait(2 * i + 1)
        acc_odd = tle.gpu.wgmma(odd.slot.a, odd.slot.b, acc_odd, trans_b=TRANS_B)
        acc_tail_odd = tle.gpu.wgmma(
            odd.slot.a, odd.slot.b_tail, acc_tail_odd, trans_b=TRANS_B
        )
        acc = tle.gpu.wgmma_wait(0, acc)
        acc_tail = tle.gpu.wgmma_wait(0, acc_tail)
        acc_odd = tle.gpu.wgmma_wait(0, acc_odd)
        acc_tail_odd = tle.gpu.wgmma_wait(0, acc_tail_odd)
        reader.release(2 * i)
        reader.release(2 * i + 1)
    acc += acc_odd
    acc_tail += acc_tail_odd
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * (BLOCK_N + BLOCK_N_TAIL) + tl.arange(0, BLOCK_N)
    cols_tail = (
        tl.program_id(1) * (BLOCK_N + BLOCK_N_TAIL)
        + BLOCK_N
        + tl.arange(0, BLOCK_N_TAIL)
    )
    if SPLIT_K > 1:
        C += tl.program_id(2) * M * N
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(
        C + rows[:, None] * N + cols_tail[None, :],
        acc_tail,
        (rows[:, None] < M) & (cols_tail[None, :] < N),
    )


@libentry()
@triton.jit
def mm_ilp_stream_kernel(
    A,
    B,
    B_tail,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_TAIL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    TRANS_B: tl.constexpr,
    CONSUMER_WARPS: tl.constexpr,
    CACHE_A: tl.constexpr,
    DTYPE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    LEADER_ONLY: tl.constexpr = False,
    ROTATE_K: tl.constexpr = 0,
):
    a = tle.gpu.alloc((STAGES, BLOCK_M, BLOCK_K), A.dtype)
    if TRANS_B:
        b = tle.gpu.alloc((STAGES, BLOCK_N, BLOCK_K), B.dtype)
        b_tail = tle.gpu.alloc((STAGES, BLOCK_N_TAIL, BLOCK_K), B_tail.dtype)
    else:
        b = tle.gpu.alloc((STAGES, BLOCK_K, BLOCK_N), B.dtype)
        b_tail = tle.gpu.alloc((STAGES, BLOCK_K, BLOCK_N_TAIL), B_tail.dtype)
    pipe = tle.pipe(capacity=STAGES, a=a, b=b, b_tail=b_tail)
    tle.gpu.warp_specialize(
        (
            (
                mm_ilp_stream_consumer,
                (
                    C,
                    pipe.reader(),
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_N_TAIL,
                    BLOCK_K,
                    STAGES,
                    TRANS_B,
                    SPLIT_K,
                ),
            ),
            (
                mm_ilp_stream_producer,
                (
                    A,
                    B,
                    B_tail,
                    pipe.writer(),
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_N_TAIL,
                    BLOCK_K,
                    STAGES,
                    TRANS_B,
                    CONSUMER_WARPS,
                    CACHE_A,
                    SPLIT_K,
                    LEADER_ONLY,
                    ROTATE_K,
                ),
            ),
        ),
        (4,),
        (64,),
    )


@triton.jit
def mm_persistent_tile(
    pid,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    group_width = GROUP_M * tl.cdiv(N, BLOCK_N)
    group_id = pid // group_width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    within_group = pid % group_width
    return (
        group_id * GROUP_M + within_group % group_size,
        within_group // group_size,
    )


@triton.jit
def mm_persistent_tme_producer(
    a_desc,
    b_desc,
    writer,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    TRANS_B: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    A_OUTER: tl.constexpr = "persist",
    B_OUTER: tl.constexpr = "normal",
):
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    tiles: tl.constexpr = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    count = tl.cdiv(tiles - tl.program_id(0), NUM_CTAS) * k_tiles
    for i in range(count):
        pid = tl.program_id(0) + i // k_tiles * NUM_CTAS
        pid_m, pid_n = mm_persistent_tile(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        start_m = ROW_OFFSET + pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N
        start_k = i % k_tiles * BLOCK_K
        slot = writer.acquire(i)
        tle.gpu.copy(
            a_desc,
            slot.a,
            (BLOCK_M, BLOCK_K),
            (start_m, start_k),
            cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="normal", outer=A_OUTER),
        )
        if TRANS_B:
            tle.gpu.copy(
                b_desc,
                slot.b,
                (BLOCK_N, BLOCK_K),
                (start_n, start_k),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="normal", outer=B_OUTER
                ),
            )
        else:
            tle.gpu.copy(
                b_desc,
                slot.b,
                (BLOCK_K, BLOCK_N),
                (start_k, start_n),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="normal", outer=B_OUTER
                ),
            )
        writer.commit(i)


@triton.jit
def mm_persistent_tme_consumer(
    C,
    reader,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    TRANS_B: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)
    tiles: tl.constexpr = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    for pid in range(tl.program_id(0), tiles, NUM_CTAS):
        offset = (pid - tl.program_id(0)) // NUM_CTAS * k_tiles
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k in range(k_tiles):
            i = offset + k
            result = reader.wait(i)
            acc = tle.gpu.wgmma(result.slot.a, result.slot.b, acc, trans_b=TRANS_B)
            acc = tle.gpu.wgmma_wait(0, acc)
            if k < k_tiles - STAGES:
                reader.release(i)
        pid_m, pid_n = mm_persistent_tile(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(
            C + (ROW_OFFSET + rm[:, None]) * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )
        # The output conversion reuses shared memory before the next tile.
        for end_slot in tl.static_range(STAGES):
            step = offset + k_tiles - STAGES + end_slot
            reader.release(step)


@libentry()
@triton.jit
def mm_persistent_tme_kernel(
    a_desc,
    b_desc,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    TRANS_B: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    DTYPE: tl.constexpr,
    ROW_OFFSET: tl.constexpr = 0,
    A_OUTER: tl.constexpr = "persist",
    B_OUTER: tl.constexpr = "normal",
):
    a = tle.gpu.alloc((STAGES, BLOCK_M, BLOCK_K), a_desc.dtype)
    if TRANS_B:
        b = tle.gpu.alloc((STAGES, BLOCK_N, BLOCK_K), b_desc.dtype)
    else:
        b = tle.gpu.alloc((STAGES, BLOCK_K, BLOCK_N), b_desc.dtype)
    pipe = tle.pipe(capacity=STAGES, a=a, b=b)
    tle.gpu.warp_specialize(
        (
            (
                mm_persistent_tme_consumer,
                (
                    C,
                    pipe.reader(),
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    STAGES,
                    TRANS_B,
                    GROUP_M,
                    NUM_CTAS,
                    ROW_OFFSET,
                ),
            ),
            (
                mm_persistent_tme_producer,
                (
                    a_desc,
                    b_desc,
                    pipe.writer(),
                    M,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    STAGES,
                    TRANS_B,
                    NUM_WARPS,
                    GROUP_M,
                    NUM_CTAS,
                    ROW_OFFSET,
                    A_OUTER,
                    B_OUTER,
                ),
            ),
        ),
        (4,),
        (64,),
    )


# A single compute partition owns both output panels; a separate partition
# loads their shared-memory buffers while the previous K tile is consumed.
@triton.jit
def mm_single_tme_producer(
    A,
    B,
    T,
    writer,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    GROUP_M: tl.constexpr,
    TAIL_M: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    A_OUTER: tl.constexpr = "persist",
    B_OUTER: tl.constexpr = "normal",
):
    BLOCK_M: tl.constexpr = 256 + BLOCK_TAIL if TAIL_M else 256
    BLOCK_N: tl.constexpr = 256 if TAIL_M else 256 + BLOCK_TAIL
    pm, pn = mm_persistent_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    start_m = pm * BLOCK_M
    start_n = pn * BLOCK_N
    for i in range(tl.cdiv(K, 64)):
        slot = writer.acquire(i)
        tle.gpu.copy(
            A,
            slot.a,
            (256, 64),
            (start_m, i * 64),
            cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="normal", outer=A_OUTER),
        )
        if TRANS_B:
            tle.gpu.copy(
                B,
                slot.b,
                (256, 64),
                (start_n, i * 64),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="normal", outer=B_OUTER
                ),
            )
        else:
            tle.gpu.copy(
                B,
                slot.b,
                (64, 256),
                (i * 64, start_n),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="normal", outer=B_OUTER
                ),
            )
        if TAIL_M:
            tle.gpu.copy(
                T,
                slot.tail,
                (BLOCK_TAIL, 64),
                (start_m + 256, i * 64),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="normal", outer=A_OUTER
                ),
            )
        elif TRANS_B:
            tle.gpu.copy(
                T,
                slot.tail,
                (BLOCK_TAIL, 64),
                (start_n + 256, i * 64),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="normal", outer=B_OUTER
                ),
            )
        else:
            tle.gpu.copy(
                T,
                slot.tail,
                (64, BLOCK_TAIL),
                (i * 64, start_n + 256),
                cache_policy=tle.gpu.mthreads.TMECachePolicy(
                    inner="normal", outer=B_OUTER
                ),
            )
        writer.commit(i)


@triton.jit
def mm_single_sqmma_consumer(
    C,
    reader,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    GROUP_M: tl.constexpr,
    TAIL_M: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    acc = tl.zeros((256, 256), tl.float32)
    if TAIL_M:
        acc_tail = tl.zeros((BLOCK_TAIL, 256), tl.float32)
    else:
        acc_tail = tl.zeros((256, BLOCK_TAIL), tl.float32)
    for i in range(tl.cdiv(K, 64)):
        result = reader.wait(i)
        acc = tle.gpu.wgmma(result.slot.a, result.slot.b, acc, trans_b=TRANS_B)
        if TAIL_M:
            acc_tail = tle.gpu.wgmma(
                result.slot.tail, result.slot.b, acc_tail, trans_b=TRANS_B
            )
        else:
            acc_tail = tle.gpu.wgmma(
                result.slot.a, result.slot.tail, acc_tail, trans_b=TRANS_B
            )
        acc = tle.gpu.wgmma_wait(0, acc)
        acc_tail = tle.gpu.wgmma_wait(0, acc_tail)
        reader.release(i)

    BLOCK_M: tl.constexpr = 256 + BLOCK_TAIL if TAIL_M else 256
    BLOCK_N: tl.constexpr = 256 if TAIL_M else 256 + BLOCK_TAIL
    pm, pn = mm_persistent_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    rows = pm * BLOCK_M + tl.arange(0, 256)
    cols = pn * BLOCK_N + tl.arange(0, 256)
    tl.store(
        C + rows[:, None].to(tl.int64) * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    if TAIL_M:
        rows_tail = pm * BLOCK_M + 256 + tl.arange(0, BLOCK_TAIL)
        tl.store(
            C + rows_tail[:, None].to(tl.int64) * N + cols[None, :],
            acc_tail,
            (rows_tail[:, None] < M) & (cols[None, :] < N),
        )
    else:
        cols_tail = pn * BLOCK_N + 256 + tl.arange(0, BLOCK_TAIL)
        tl.store(
            C + rows[:, None].to(tl.int64) * N + cols_tail[None, :],
            acc_tail,
            (rows[:, None] < M) & (cols_tail[None, :] < N),
        )


@libentry()
@triton.jit
def mm_single_consumer_kernel(
    A,
    B,
    T,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    GROUP_M: tl.constexpr,
    TAIL_M: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    A_OUTER: tl.constexpr = "persist",
    B_OUTER: tl.constexpr = "normal",
):
    a = tle.gpu.alloc((2, 256, 64), A.dtype)
    if TAIL_M:
        tail = tle.gpu.alloc((2, BLOCK_TAIL, 64), T.dtype)
    if TRANS_B:
        b = tle.gpu.alloc((2, 256, 64), B.dtype)
    else:
        b = tle.gpu.alloc((2, 64, 256), B.dtype)
    if not TAIL_M:
        if TRANS_B:
            tail = tle.gpu.alloc((2, BLOCK_TAIL, 64), T.dtype)
        else:
            tail = tle.gpu.alloc((2, 64, BLOCK_TAIL), T.dtype)
    pipe = tle.pipe(capacity=2, a=a, b=b, tail=tail)
    tle.gpu.warp_specialize(
        (
            (
                mm_single_sqmma_consumer,
                (
                    C,
                    pipe.reader(),
                    M,
                    N,
                    K,
                    TRANS_B,
                    GROUP_M,
                    TAIL_M,
                    BLOCK_TAIL,
                ),
            ),
            (
                mm_single_tme_producer,
                (
                    A,
                    B,
                    T,
                    pipe.writer(),
                    M,
                    N,
                    K,
                    TRANS_B,
                    GROUP_M,
                    TAIL_M,
                    BLOCK_TAIL,
                    A_OUTER,
                    B_OUTER,
                ),
            ),
        ),
        (4,),
        (64,),
    )


def mm_tme_narrow(a, b):
    """Use enough narrow tiles to fill the device without an extra launch."""
    m, k = a.shape
    n = b.shape[1]
    trans_b = b.stride(0) == 1
    c = torch.empty((m, n), dtype=a.dtype, device=a.device)
    stream = k >= 512 and (
        (m <= 8 and (trans_b or n > 512)) or (8 < m <= 128 and n <= 512)
    )
    if stream:
        bm, bn, bk = (32, 32, 256) if m <= 8 else (16, 64, 256)
    elif k < 512 or (m > 256 and max(m, n) <= 512):
        bm, bn, bk = 32, 64, 64
    elif m > 256:
        bm, bn, bk = 128, 256, 64
    elif m <= 8:
        bm, bn, bk = 32, 32, 256
    else:
        bm, bk = 64, 128
        bn = 64 if triton.cdiv(m, 64) * triton.cdiv(n, 64) <= 32 else 128

    a_desc = TensorDescriptor.from_tensor(a, [bm, bk])
    b_desc = TensorDescriptor.from_tensor(
        b.t() if trans_b else b, [bn, bk] if trans_b else [bk, bn]
    )
    with torch_device_fn.device(a.device):
        grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
        if stream:
            mm_stream_tme_kernel[grid](
                a_desc,
                b_desc,
                c,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                4,
                1,
                trans_b,
                4,
                1,
                0,
                a.dtype,
                CACHE_A=m > 8,
                LEADER_ONLY=True,
                num_warps=4,
                num_stages=1,
                enable_backend_opt=True,
            )
        else:
            mm_pipe_kernel[grid](
                a_desc,
                b_desc,
                c,
                m,
                n,
                k,
                trans_b,
                bn,
                BLOCK_M=bm,
                BLOCK_K=bk,
                CAPACITY=4,
                num_warps=8 if bm == 128 else 4,
                num_stages=1,
                enable_backend_opt=True,
            )
    return c


def mm_stream_descriptor_pre_hook(args):
    bm, bn, bk = args["BLOCK_M"], args["BLOCK_N"], args["BLOCK_K"]
    args["A"].block_shape = [bm, bk]
    args["B"].block_shape = [bn, bk] if args["TRANS_B"] else [bk, bn]


def mm_stream_configs():
    return [
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_K": 128,
                "STAGES": stages,
                "LEADER_ONLY": leader,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=mm_stream_descriptor_pre_hook,
        )
        for bm, bn, stages, leader in (
            (32, 64, 3, True),
            (16, 64, 2, True),
            (16, 64, 2, False),
            (32, 64, 2, False),
            (32, 64, 3, False),
            (32, 64, 4, True),
            (32, 128, 2, True),
            (32, 128, 2, False),
            (32, 128, 3, True),
        )
    ]


def mm_stream_prune(configs, args, **kwargs):
    args = {**args, **kwargs}
    n, k, trans_b = args["N"], args["K"], args["TRANS_B"]
    selected = []
    for config in configs:
        meta = config.kwargs
        if trans_b:
            if n >= 32768:
                if (
                    meta["BLOCK_M"],
                    meta["BLOCK_N"],
                    meta["STAGES"],
                    meta["LEADER_ONLY"],
                ) != (32, 128, 3, True):
                    continue
            elif n > 8192 and meta["BLOCK_M"] < 32:
                continue
            elif not meta["LEADER_ONLY"] and (
                n > 4096 or (k > 8192 and args["SPLIT_K"] == 1)
            ):
                continue
        else:
            # Keep NN on the measured geometry choices: wider columns reuse A,
            # while narrow outputs and long reductions use more CTAs. Expanding
            # the NN candidate set exposed async-barrier failures during tuning.
            bn = 64 if n < 4096 or k >= 8192 else 128
            stages = 3 if n < 4096 else 2
            if (
                meta["BLOCK_M"] != 32
                or meta["BLOCK_N"] != bn
                or meta["STAGES"] != stages
                or meta["LEADER_ONLY"]
            ):
                continue
        selected.append(config)
    return selected


mm_stream_tuned_kernel = libentry()(
    libtuner(
        configs=mm_stream_configs(),
        key=["M", "N", "K", "TRANS_B", "DTYPE", "SPLIT_K"],
        strategy="default",
        prune_configs_by={"early_config_prune": mm_stream_prune},
        warmup=5,
        rep=15,
    )(mm_stream_tme_kernel.fn)
)


def mm_tme_descriptor_pre_hook(args):
    bm, bn, bk = args["BLOCK_M"], args["BLOCK_N"], args["BLOCK_K"]
    args["A"].block_shape = [bm, bk]
    args["B"].block_shape = [bn, bk] if args["TRANS_B"] else [bk, bn]
    tail = args["BLOCK_TAIL"] or 32
    args["B_tail"].block_shape = [tail, bk] if args["TRANS_B"] else [bk, tail]


def mm_tme_configs():
    configs = []
    # Medium output grids trade row reuse against CTA parallelism. Keep the
    # independent-accumulator algorithm in the same measured config space.
    for algorithm, bm, bn, tail, bk, stages, warps, cache_a, leader, rotate in (
        (1, 64, 64, 0, 64, 4, 4, True, False, 0),
        (1, 128, 256, 0, 64, 4, 16, True, True, 0),
        (1, 256, 256, 0, 32, 4, 16, True, False, 0),
        (1, 256, 256, 0, 64, 2, 16, True, False, 0),
        (1, 64, 256, 0, 128, 2, 8, False, False, 0),
        (1, 128, 256, 0, 64, 4, 8, False, False, 0),
        (1, 128, 256, 0, 128, 2, 16, True, False, 0),
        (2, 64, 128, 32, 64, 5, 8, True, True, 0),
        (2, 64, 128, 32, 64, 5, 8, True, True, 11),
        (2, 128, 64, 16, 64, 7, 8, True, True, 0),
        (2, 64, 128, 32, 64, 4, 8, True, True, 0),
        (2, 128, 128, 128, 64, 3, 16, True, False, 0),
        (2, 128, 64, 64, 64, 4, 8, True, False, 0),
        (2, 64, 256, 64, 64, 4, 8, True, False, 0),
        (2, 128, 256, 64, 64, 3, 16, True, False, 0),
        (2, 128, 128, 32, 64, 4, 16, True, False, 0),
        (2, 128, 64, 64, 64, 6, 8, True, False, 0),
        (2, 128, 128, 64, 64, 4, 8, True, False, 0),
        (3, 256, 256, 64, 64, 2, 16, False, False, 0),
    ):
        configs.append(
            triton.Config(
                {
                    "ALGORITHM": algorithm,
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_TAIL": tail,
                    "BLOCK_K": bk,
                    "STAGES": stages,
                    "CACHE_A": cache_a,
                    "LEADER_ONLY": leader,
                    "ROTATE_K": rotate,
                    "CONSUMER_WARPS": warps,
                },
                num_warps=warps,
                num_stages=1,
                pre_hook=mm_tme_descriptor_pre_hook,
            )
        )
    return configs


def mm_tme_prune(configs, args, **kwargs):
    m, n = args["M"], args["N"]
    trans_b, split_k = args["TRANS_B"], args["SPLIT_K"]
    num_sms = args["NUM_SMS"]
    selected = []
    for config in configs:
        meta = config.kwargs
        algorithm = meta["ALGORITHM"]
        bm, bn, bk = meta["BLOCK_M"], meta["BLOCK_N"], meta["BLOCK_K"]
        tail = meta["BLOCK_TAIL"]
        if split_k > 1:
            if algorithm == 1 and bn == 256 and bm in (64, 128):
                selected.append(config)
            elif algorithm == 2 and bm == 128 and bn == 128 and tail == 32:
                selected.append(config)
        elif n < 8192:
            if algorithm == 1 and (
                bm == bn == bk == 64
                or (bm == 128 and bn == 256 and config.num_warps == 16 and bk == 64)
            ):
                selected.append(config)
            elif algorithm == 2 and config.num_warps == 8:
                if (tail < 32 or meta["ROTATE_K"]) and not trans_b:
                    continue
                selected.append(config)
        else:
            # Compare the occupied fraction of device-sized waves. This selects the
            # measured 256- or 320-column family without enumerating shapes.
            tiles256, tiles320 = triton.cdiv(n, 256), triton.cdiv(n, 320)
            wide = tiles320 * triton.cdiv(tiles256, num_sms) >= tiles256 * triton.cdiv(
                tiles320, num_sms
            )
            if m <= 128:
                target = (256, 64) if wide else (128, 128)
                if (
                    algorithm == 2
                    and bm == 128
                    and (bn, tail) == target
                    and meta["STAGES"] == 3
                ):
                    selected.append(config)
            else:
                # This transposed-weight cache regime retains its measured
                # stream configuration; the wider candidates are not general.
                if trans_b and args["K"] == 3584 and n >= 128256:
                    wide = False
                if wide:
                    if algorithm == 3:
                        selected.append(config)
                elif (
                    algorithm == 1
                    and bm == bn == 256
                    and (bk, meta["STAGES"])
                    == ((32, 4) if n <= 16384 and not trans_b else (64, 2))
                ):
                    selected.append(config)
    return selected


# Resolve LibEntry.fn outside JIT: dependency analysis cannot copy LibEntry.
mm_stream_tme_device = mm_stream_tme_kernel.fn
mm_ilp_stream_device = mm_ilp_stream_kernel.fn
mm_wide_tle_device = mm_wide_tle_kernel.fn


@libentry()
@libtuner(
    configs=mm_tme_configs(),
    key=["M", "N", "K", "TRANS_B", "DTYPE", "SPLIT_K", "NUM_SMS"],
    strategy="default",
    prune_configs_by={"early_config_prune": mm_tme_prune},
    warmup=5,
    rep=15,
)
@triton.jit
def mm_tme_kernel(
    A,
    B,
    B_tail,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    DTYPE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    ALGORITHM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    LEADER_ONLY: tl.constexpr,
    CACHE_A: tl.constexpr,
    ROTATE_K: tl.constexpr,
    CONSUMER_WARPS: tl.constexpr,
):
    if ALGORITHM == 1:
        mm_stream_tme_device(
            A,
            B,
            C,
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            STAGES,
            SPLIT_K,
            TRANS_B,
            CONSUMER_WARPS,
            1,
            0,
            DTYPE,
            CACHE_A,
            LEADER_ONLY,
        )
    elif ALGORITHM == 2:
        mm_ilp_stream_device(
            A,
            B,
            B_tail,
            C,
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_TAIL,
            BLOCK_K,
            STAGES,
            TRANS_B,
            CONSUMER_WARPS,
            CACHE_A,
            DTYPE,
            SPLIT_K,
            LEADER_ONLY,
            ROTATE_K,
        )
    else:
        mm_wide_tle_device(
            A,
            B,
            B_tail,
            C,
            M,
            N,
            K,
            TRANS_B,
            BLOCK_K,
            STAGES,
            SPLIT_K=SPLIT_K,
            DTYPE=DTYPE,
            BLOCK_TAIL=BLOCK_TAIL,
            GROUP_M=0,
        )


def mm_tme(a, b):
    m, k = a.shape
    n = b.shape[1]
    if n <= 1024:
        return mm_tme_narrow(a, b)
    trans_b = b.stride(0) == 1
    # A long reduction with fewer than two column waves benefits from Split-K.
    if m <= 8:
        split_k = 2 if trans_b and k >= 16384 and n <= 8192 else 1
    else:
        split_k = 2 + int(trans_b and k < 4 * n) if k >= 8192 and n <= 8192 else 1
    c = torch.empty((m, n), dtype=a.dtype, device=a.device)
    partial = (
        torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
        if split_k > 1
        else c
    )
    a_desc = TensorDescriptor(a, [m, k], [k, 1], [1, 1])
    b_shape = [n, k] if trans_b else [k, n]
    b_strides = [k, 1] if trans_b else [n, 1]
    b_desc = TensorDescriptor(b, b_shape, b_strides, [1, 1])
    with torch_device_fn.device(a.device):
        if m <= 8:
            mm_stream_tuned_kernel[
                lambda meta: (
                    triton.cdiv(m, meta["BLOCK_M"]),
                    triton.cdiv(n, meta["BLOCK_N"]),
                    split_k,
                )
            ](
                a_desc,
                b_desc,
                partial,
                m,
                n,
                k,
                SPLIT_K=split_k,
                TRANS_B=trans_b,
                CONSUMER_WARPS=4,
                INNER_CACHE=1,
                OUTER_CACHE=0,
                DTYPE=a.dtype,
                CACHE_A=False,
                enable_backend_opt=True,
            )
        else:
            num_sms = torch.musa.get_device_properties(
                a.device.index
            ).multi_processor_count
            b_tail_desc = TensorDescriptor(b, b_shape, b_strides, [1, 1])
            mm_tme_kernel[
                lambda meta: (
                    triton.cdiv(m, meta["BLOCK_M"]),
                    triton.cdiv(n, meta["BLOCK_N"] + meta["BLOCK_TAIL"]),
                    split_k,
                )
            ](
                a_desc,
                b_desc,
                b_tail_desc,
                partial,
                m,
                n,
                k,
                trans_b,
                a.dtype,
                split_k,
                num_sms,
                enable_backend_opt=True,
            )
        if split_k > 1:
            mm_splitk_reduce_kernel[(triton.cdiv(m * n, 256),)](
                partial, c, m * n, split_k, 256, num_warps=4
            )
    return c


def _mm_large_descriptor_hook(nargs):
    bk = nargs["BLOCK_K"]
    tail = nargs["BLOCK_TAIL"]
    nt = nargs["TRANS_B"]
    nargs["a_desc"].block_shape = [256, bk]
    nargs["b_desc"].block_shape = [256, bk] if nt else [bk, 256]
    nargs["b_tail_desc"].block_shape = [tail, bk] if nt else [bk, tail]


# Reuse the existing TLE kernel; only validated shared-memory sizes enter tuning.
mm_large_tle_tuned = libentry()(
    libtuner(
        configs=[
            triton.Config(
                {"BLOCK_K": bk, "STAGES": stages, "BLOCK_TAIL": tail, "GROUP_M": 4},
                num_warps=16,
                num_stages=1,
                pre_hook=_mm_large_descriptor_hook,
            )
            for bk, stages, tail in ((64, 2, 64), (32, 4, 64), (32, 5, 32))
        ],
        key=["M", "N", "K", "TRANS_B", "DTYPE"],
        strategy=["default"] * 5,
        warmup=5,
        rep=5,
    )(mm_wide_tle_kernel.fn)
)


def mm_large_half(a, b, M, N, K, trans_b):
    """Launch an aligned, row-major-A half GEMM from the large-output family."""
    dtype = a.dtype
    num_sms = torch.musa.get_device_properties(a.device.index).multi_processor_count
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # Keep the measured wave decomposition: tile lifetime spans the C epilogue.
    if M == 8192 and N == K == 3584:
        a_desc = TensorDescriptor(a, [M, K], [K, 1], [256, 64])
        b_shape = [N, K] if trans_b else [K, N]
        b_strides = [K, 1] if trans_b else [N, 1]
        b_desc = TensorDescriptor(
            b, b_shape, b_strides, [256, 64] if trans_b else [64, 256]
        )
        with torch_device_fn.device(a.device):
            num_ctas = (
                triton.cdiv(7680, 256) * triton.cdiv(N, 256) if trans_b else num_sms
            )
            gm = 4 if trans_b and dtype == torch.bfloat16 else 2
            mm_persistent_tme_kernel[(num_ctas,)](
                a_desc,
                b_desc,
                c,
                7680,
                N,
                K,
                256,
                256,
                64,
                3,
                trans_b,
                16,
                gm,
                num_ctas,
                dtype,
                num_warps=16,
                num_stages=1,
                enable_backend_opt=True,
            )
            if trans_b:
                bm, bn = 128, 256
                bk, stages, warps = (
                    (128, 2, 8) if dtype == torch.float16 else (64, 4, 16)
                )
            else:
                bm, bn, bk, stages, warps = 256, 128, 64, 4, 16
            a_tail = TensorDescriptor(a, [M, K], [K, 1], [bm, bk])
            b_tail = TensorDescriptor(
                b, b_shape, b_strides, [bn, bk] if trans_b else [bk, bn]
            )
            # One CTA per output tile in the remaining rows.
            tail_rows = M - 7680
            tail_tiles = triton.cdiv(tail_rows, bm) * triton.cdiv(N, bn)
            mm_persistent_tme_kernel[(tail_tiles,)](
                a_tail,
                b_tail,
                c,
                tail_rows,
                N,
                K,
                bm,
                bn,
                bk,
                stages,
                trans_b,
                warps,
                2,
                tail_tiles,
                dtype,
                ROW_OFFSET=7680,
                num_warps=warps,
                num_stages=1,
                enable_backend_opt=True,
            )
        return c

    # Reuse native TLE kernels for the former unified-raw dispatch set.
    if (
        M == 8192
        and K == 4096
        and trans_b
        and (N == 28672 or (dtype == torch.bfloat16 and N in (6144, 14336)))
    ):
        a_outer, b_outer = (
            ("persist", "normal") if N == 14336 else ("normal", "persist")
        )
        if N == 6144:
            a_desc = TensorDescriptor(a, [M, K], [K, 1], [256, 32])
            b_desc = TensorDescriptor(b, [N, K], [K, 1], [256, 32])
            with torch_device_fn.device(a.device):
                mm_persistent_tme_kernel[(num_sms,)](
                    a_desc,
                    b_desc,
                    c,
                    M,
                    N,
                    K,
                    256,
                    256,
                    32,
                    4,
                    True,
                    16,
                    2,
                    num_sms,
                    dtype,
                    A_OUTER=a_outer,
                    B_OUTER=b_outer,
                    num_warps=16,
                    num_stages=1,
                    enable_backend_opt=True,
                )
        else:
            a_desc = TensorDescriptor(a, [M, K], [K, 1], [256, 64])
            b_desc = TensorDescriptor(b, [N, K], [K, 1], [256, 64])
            tail_desc = TensorDescriptor(b, [N, K], [K, 1], [64, 64])
            with torch_device_fn.device(a.device):
                mm_single_consumer_kernel[(triton.cdiv(M, 256) * triton.cdiv(N, 320),)](
                    a_desc,
                    b_desc,
                    tail_desc,
                    c,
                    M,
                    N,
                    K,
                    True,
                    4,
                    TAIL_M=False,
                    BLOCK_TAIL=64,
                    A_OUTER=a_outer,
                    B_OUTER=b_outer,
                    num_warps=16,
                    num_stages=1,
                    enable_backend_opt=True,
                )
        return c

    # One compute partition handles the main tile and its extra row/column panel.
    tall_tile = N == 4096 and (M == K == 4096 or (M == 8192 and K == 14336))
    wide_tile = M == N == K == 2048 or (M == 8192 and K == 4096 and N in (4096, 128256))
    if tall_tile or wide_tile:
        a_desc = TensorDescriptor(a, [M, K], [K, 1], [256, 64])
        b_shape = [N, K] if trans_b else [K, N]
        b_strides = [K, 1] if trans_b else [N, 1]
        b_desc = TensorDescriptor(
            b, b_shape, b_strides, [256, 64] if trans_b else [64, 256]
        )
        if tall_tile:
            tail = 32 if K <= 4096 else 64
            bm, bn, gm = 256 + tail, 256, 2
            tail_desc = TensorDescriptor(a, [M, K], [K, 1], [tail, 64])
        else:
            tail = 64
            bm, bn = 256, 256 + tail
            gm = (
                2
                if M == 8192 and N == 4096 and dtype == torch.float16 and not trans_b
                else 4
            )
            tail_desc = TensorDescriptor(b, b_shape, b_strides, [64, 64])
        with torch_device_fn.device(a.device):
            mm_single_consumer_kernel[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
                a_desc,
                b_desc,
                tail_desc,
                c,
                M,
                N,
                K,
                trans_b,
                gm,
                TAIL_M=tall_tile,
                BLOCK_TAIL=tail,
                num_warps=16,
                num_stages=1,
                enable_backend_opt=True,
            )
        return c

    # Narrow output: share B across a 256-row tile and its extra row panel.
    if N <= 1024 and N % 256 == 0 and M >= 4 * N and M * N * a.element_size() < 2**31:
        tail, gm = (64, 2) if N <= 512 else (32, 1)
        a_desc = TensorDescriptor(a, [M, K], [K, 1], [256, 32])
        a_tail = TensorDescriptor(a, [M, K], [K, 1], [tail, 32])
        b_desc = TensorDescriptor(
            b,
            [N, K] if trans_b else [K, N],
            [K, 1] if trans_b else [N, 1],
            [256, 32] if trans_b else [32, 256],
        )
        with torch_device_fn.device(a.device):
            mm_tall_tle_kernel[(triton.cdiv(M, 256 + tail) * triton.cdiv(N, 256),)](
                a_desc,
                a_tail,
                b_desc,
                c,
                M,
                N,
                K,
                trans_b,
                tail,
                gm,
                num_warps=16,
                num_stages=1,
                enable_backend_opt=True,
            )
        return c

    # Long reductions use split-K only when output tiles cannot fill the machine.
    long_k = K >= 16384 and K >= 4 * N
    split = 1
    if long_k:
        output_tiles = triton.cdiv(M, 256) * triton.cdiv(N, 320)
        if K >= 65536 and output_tiles < num_sms:
            split = min(32, triton.cdiv(4 * num_sms, output_tiles))
        tail, stages, gm = (64, 4, 0) if split > 1 else (32, 5, 4)
        bk = 32
    else:
        bk, tail = 64, 64

    a_desc = TensorDescriptor(a, [M, K], [K, 1], [256, bk])
    b_shape = [N, K] if trans_b else [K, N]
    b_strides = [K, 1] if trans_b else [N, 1]
    b_desc = TensorDescriptor(
        b, b_shape, b_strides, [256, bk] if trans_b else [bk, 256]
    )
    b_tail = TensorDescriptor(
        b, b_shape, b_strides, [tail, bk] if trans_b else [bk, tail]
    )
    partial = (
        torch.empty((split, M, N), device=a.device, dtype=torch.float32)
        if split > 1
        else c
    )
    with torch_device_fn.device(a.device):
        if long_k:
            grid = (
                (triton.cdiv(M, 256), triton.cdiv(N, 320), split)
                if split > 1
                else (triton.cdiv(M, 256) * triton.cdiv(N, 288),)
            )
            mm_wide_tle_kernel[grid](
                a_desc,
                b_desc,
                b_tail,
                partial,
                M,
                N,
                K,
                trans_b,
                bk,
                stages,
                SPLIT_K=split,
                DTYPE=dtype,
                BLOCK_TAIL=tail,
                GROUP_M=gm,
                num_warps=16,
                num_stages=1,
                enable_backend_opt=True,
            )
        else:
            mm_large_tle_tuned[
                lambda meta: (
                    triton.cdiv(M, 256) * triton.cdiv(N, 256 + meta["BLOCK_TAIL"]),
                )
            ](
                a_desc,
                b_desc,
                b_tail,
                c,
                M,
                N,
                K,
                trans_b,
                SPLIT_K=1,
                DTYPE=dtype,
                enable_backend_opt=True,
            )
        if split > 1:
            mm_splitk_reduce_kernel[(triton.cdiv(M * N, 256),)](
                partial, c, M * N, split, BLOCK=256
            )
    return c


def mm(a, b):
    a_dtype, b_dtype = a.dtype, b.dtype
    M, K = a.shape
    N = b.shape[1]
    if N == 1:
        c = torch.empty((M, N), device=a.device, dtype=a_dtype)
        return gemv_mm(a, b, c, M, K)
    if a_dtype == b_dtype == torch.float32:
        return mm_fp32(a, b)

    if a_dtype == b_dtype and a_dtype in (torch.float16, torch.bfloat16):
        if a.is_contiguous() and is_supported_sqmma_layout(b):
            trans_b = b.stride(0) == 1
            small_rows = (
                0 < M <= 256
                and N >= 256
                and N % 32 == 0
                and 512 <= K <= 32768
                and K % 128 == 0
            )
            compact_output = (
                0 < M <= 1024
                and 32 <= N <= 1024
                and N % 32 == 0
                and 64 <= K <= 4096
                and K % 64 == 0
                and (K < 512 or M > 256)
            )
            if small_rows or compact_output:
                # With very few transposed-weight rows, a SIMT reduction avoids
                # TME descriptor and producer/consumer setup.
                if (
                    trans_b
                    and N <= 512
                    and 2048 <= K <= 4096
                    and (M == 1 or (a_dtype == torch.float16 and M <= 3))
                ):
                    return gemv_nt_mm(a, b, N, K)
                return mm_tme(a, b)
            if M > 1024 and N >= 256 and N % 64 == 0 and K > 0 and K % 32 == 0:
                # Retain the upstream TLE pipeline for its measured NN wins.
                if (
                    M == 8192
                    and not trans_b
                    and ((N == 6144 and K == 4096) or (N == 3584 and K == 18944))
                ):
                    return mm_sqmma_ws(a, b, M, N, K)
                return mm_large_half(a, b, M, N, K, trans_b)

        # Split-K supplies parallelism for tiny outputs, including [T,3584] x
        # [3584,64]. Bound its workspace and amortize the separate reduction.
        if (
            M > 0
            and N > 1
            and 512 <= K <= 32768
            and triton.cdiv(M, 32) * triton.cdiv(N, 32) <= min(64, K // 16)
        ):
            return mm_small(a, b)

    if is_sqmma_compatible(a, b, N, K):
        return mm_sqmma(a, b, M, N, K)
    return mm_fma(a, b)
