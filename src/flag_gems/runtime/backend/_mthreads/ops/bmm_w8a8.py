# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.ops.bmm_w8a8 import bmm_w8a8 as generic_bmm_w8a8
from flag_gems.ops.bmm_w8a8 import validate_inputs
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

try:
    from triton._C.libtriton.mthreads import SUPPORTS_TLE_WGMMA_INT8
except ImportError:
    SUPPORTS_TLE_WGMMA_INT8 = False

from triton.experimental.tle import language as tle

from .mm_w8a8_fp8 import _produce as produce_tiles

logger = logging.getLogger(__name__)

GEMV_MAX_ROWS = 4
GEMV_MIN_REDUCTION = 256
GEMV_BLOCK_COLUMNS = 4
GEMV_BLOCK_REDUCTION = 1024
GEMV_WARPS = 4
LARGE_MATRIX_ROWS = 1024
MEDIUM_MATRIX_ROWS = 256
SMALL_MATRIX_ROWS = 64
# BM, BN, BK, warps, compiler stages, rasterization group.
TINY_TILE_CONFIG = (16, 64, 64, 4, 2, 1)
SMALL_TILE_CONFIG = (64, 64, 64, 4, 2, 1)
MEDIUM_TILE_CONFIG = (128, 128, 64, 4, 2, 1)
ALIGNED_TILE_CONFIG = (256, 64, 64, 4, 2, 1)
# Host tensor-descriptor BM, BN, BK; one stage and four warps stay fixed.
HOST_TME_TILE_CONFIG = (32, 32, 512)
TLE_SMALL_MAX_ROWS = 128
TLE_MEDIUM_MAX_ROWS = 512
# BM, BN, BK, pipe slots; main and producer partitions each use four warps.
TLE_SMALL_TILE_CONFIG = (32, 64, 256, 3)
TLE_MEDIUM_TILE_CONFIG = (64, 64, 256, 2)
TLE_LARGE_TILE_CONFIG = (128, 128, 128, 3)


@libentry()
@triton.jit
def channel_gemv_kernel(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    n = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    k = tl.arange(0, BK)
    total = tl.zeros((BN,), dtype=tl.int32)
    for start in range(tl.cdiv(K, BK)):
        current_k = start.to(tl.int64) * BK + k
        av = tl.load(A + batch * AB + row * K + current_k, current_k < K, other=0)
        bv = tl.load(
            B + batch * BB + n[:, None] * K + current_k[None, :],
            (n[:, None] < N) & (current_k[None, :] < K),
            other=0,
        )
        total += tl.sum(av[None, :].to(tl.int32) * bv.to(tl.int32), axis=1)
    sa = tl.load(SA + batch * M + row)
    sb = tl.load(SB + batch * N + n, n < N, other=0)
    result = total.to(tl.float32) * sa * sb
    tl.store(C + batch * M * N + row * N + n, result, n < N)


@libentry()
@triton.jit
def channel_host_tme_kernel(
    activation_descriptor,
    weight_descriptor,
    output,
    activation_scale,
    weight_scale,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    row_start = (tile // tl.cdiv(N, BN)) * BM
    column_start = (tile % tl.cdiv(N, BN)) * BN
    accumulator = tl.zeros((BM, BN), dtype=tl.int32)
    for block in tl.range(0, tl.cdiv(K, BK), num_stages=STAGES):
        activation_tile = tl.load_tensor_descriptor(
            activation_descriptor, [batch * M + row_start, block * BK]
        )
        weight_tile = tl.load_tensor_descriptor(
            weight_descriptor, [batch * N + column_start, block * BK]
        )
        accumulator = tl.dot(
            activation_tile, tl.trans(weight_tile), accumulator, out_dtype=tl.int32
        )
    rows = row_start + tl.arange(0, BM)
    columns = column_start + tl.arange(0, BN)
    batch_offset = batch.to(tl.int64)
    row_scale = tl.load(activation_scale + batch_offset * M + rows, rows < M, other=0)
    column_scale = tl.load(
        weight_scale + batch_offset * N + columns, columns < N, other=0
    )
    scaled = accumulator.to(tl.float32) * row_scale[:, None] * column_scale[None, :]
    offsets = batch_offset * M * N + rows[:, None].to(tl.int64) * N + columns[None, :]
    tl.store(output + offsets, scaled, (rows[:, None] < M) & (columns[None, :] < N))


@triton.jit
def channel_tle_consumer(
    reader,
    output,
    activation_scale,
    weight_scale,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    rows = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    columns = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    accumulator = tl.zeros((BM, BN), dtype=tl.int32)
    for block in tl.range(0, tl.cdiv(K, BK), num_stages=1):
        ready = reader.wait(block)
        accumulator = tle.gpu.wgmma(
            ready.slot.a,
            ready.slot.b0,
            accumulator,
            out_dtype=tl.int32,
            trans_b=True,
        )
        # The slot cannot be reused while SQMMA still reads its operands.
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(block)
    row_scale = tl.load(activation_scale + batch * M + rows, rows < M, other=0)
    column_scale = tl.load(weight_scale + batch * N + columns, columns < N, other=0)
    scaled = accumulator.to(tl.float32) * row_scale[:, None] * column_scale[None, :]
    output_offsets = batch * M * N + rows[:, None].to(tl.int64) * N + columns[None, :]
    tl.store(
        output + output_offsets,
        scaled,
        (rows[:, None] < M) & (columns[None, :] < N),
    )


@libentry()
@triton.jit
def channel_tle_tme_kernel(
    activation_descriptor,
    weight_descriptor,
    output,
    activation_scale,
    weight_scale,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
):
    activation_shared = tle.gpu.alloc(
        (PIPELINE_STAGES, BM, BK),
        activation_descriptor.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    weight_shared = tle.gpu.alloc(
        (PIPELINE_STAGES, BN, BK),
        weight_descriptor.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=PIPELINE_STAGES,
        scope="cta",
        name="int8_bmm",
        readers=("r0",),
        a=activation_shared,
        b0=weight_shared,
    )
    tle.gpu.warp_specialize(
        [
            (
                channel_tle_consumer,
                (
                    pipe.reader("r0", fields=("a", "b0")),
                    output,
                    activation_scale,
                    weight_scale,
                    M,
                    N,
                    K,
                    BM,
                    BN,
                    BK,
                ),
            ),
            (
                produce_tiles,
                (
                    pipe.writer(),
                    activation_descriptor,
                    weight_descriptor,
                    M,
                    N,
                    K,
                    BM,
                    BN,
                    BK,
                    1,
                    1,
                ),
            ),
        ],
        worker_num_warps=[PRODUCER_WARPS],
        worker_num_regs=[24],
    )


def bmm_w8a8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    *,
    block_size=(128, 128, 128),
    config=None,
) -> torch.Tensor:
    """Channel-wise batched W8A8 matrix multiplication on MTT S5000.

    A has shape (batch, M, K), B has shape (batch, K, N). A_scale is
    (batch, M) or (batch, M, 1); B_scale is (batch, N) or (batch, 1, N).
    INT8 accumulates in INT32. Scales are applied in FP32 before
    casting to out_dtype.
    block_size is retained for API compatibility and is unused here.
    Small M with transposed contiguous weights uses an integer GEMV reduction.
    Aligned batch-one NT inputs use host tensor descriptors and select the TLE
    pipeline when the compiler exports integer WGMMA support. Other layouts
    reuse the shared Triton kernel.
    """
    logger.debug("GEMS_MTHREADS BMM_W8A8")
    A, B = activation, weight
    A_scale, B_scale = activation_scale, weight_scale
    out = validate_inputs(A, B, A_scale, B_scale, out_dtype, out)
    batch, M, K = A.shape
    N = B.shape[2]
    if batch * M * N == 0:
        return out
    if config is not None or not (
        A.is_contiguous()
        and (B.is_contiguous() or B.transpose(1, 2).is_contiguous())
        and out.is_contiguous()
        and A_scale.is_contiguous()
        and B_scale.is_contiguous()
        and A_scale.dtype == torch.float32
        and B_scale.dtype == torch.float32
    ):
        return generic_bmm_w8a8(
            A, B, A_scale, B_scale, out_dtype, out, block_size=block_size, config=config
        )
    if M <= GEMV_MAX_ROWS and K >= GEMV_MIN_REDUCTION and B.stride(1) == 1:
        with torch_device_fn.device(A.device):
            channel_gemv_kernel[(triton.cdiv(N, GEMV_BLOCK_COLUMNS), M, batch)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                M,
                N,
                K,
                A.stride(0),
                B.stride(0),
                GEMV_BLOCK_COLUMNS,
                GEMV_BLOCK_REDUCTION,
                num_warps=GEMV_WARPS,
                num_stages=1,
            )
        return out
    if (
        batch == 1
        and M > GEMV_MAX_ROWS
        and K > 0
        and K % 128 == 0
        and N % 16 == 0
        and B.transpose(1, 2).is_contiguous()
        and A.data_ptr() % 16 == 0
        and B.data_ptr() % 16 == 0
    ):
        if SUPPORTS_TLE_WGMMA_INT8 is True:
            if M <= TLE_SMALL_MAX_ROWS:
                bm, bn, bk, pipeline_stages = TLE_SMALL_TILE_CONFIG
            elif M <= TLE_MEDIUM_MAX_ROWS:
                bm, bn, bk, pipeline_stages = TLE_MEDIUM_TILE_CONFIG
            else:
                bm, bn, bk, pipeline_stages = TLE_LARGE_TILE_CONFIG
            kernel = channel_tle_tme_kernel
            stage_parameters = (pipeline_stages, 4)
        else:
            bm, bn, bk = HOST_TME_TILE_CONFIG
            kernel = channel_host_tme_kernel
            stage_parameters = (1,)
        activation_descriptor = TensorDescriptor(A, [batch * M, K], [K, 1], [bm, bk])
        weight_descriptor = TensorDescriptor(B, [batch * N, K], [K, 1], [bn, bk])
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn), batch)
        with torch_device_fn.device(A.device):
            kernel[grid](
                activation_descriptor,
                weight_descriptor,
                out,
                A_scale,
                B_scale,
                M,
                N,
                K,
                bm,
                bn,
                bk,
                *stage_parameters,
                num_warps=4,
                num_stages=1,
                enable_backend_opt=True,
            )
        return out
    if M >= LARGE_MATRIX_ROWS:
        if M % ALIGNED_TILE_CONFIG[0] == 0 and N % ALIGNED_TILE_CONFIG[1] == 0:
            config = ALIGNED_TILE_CONFIG
        else:
            config = SMALL_TILE_CONFIG
    elif M >= MEDIUM_MATRIX_ROWS:
        config = MEDIUM_TILE_CONFIG
    elif M >= SMALL_MATRIX_ROWS:
        config = SMALL_TILE_CONFIG
    else:
        config = TINY_TILE_CONFIG
    return generic_bmm_w8a8(
        A, B, A_scale, B_scale, out_dtype, out, block_size=block_size, config=config
    )
