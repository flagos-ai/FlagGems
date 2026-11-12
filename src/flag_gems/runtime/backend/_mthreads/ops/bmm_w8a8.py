# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.bmm_w8a8 import bmm_w8a8 as generic_bmm_w8a8
from flag_gems.ops.bmm_w8a8 import validate_inputs
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

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
    Other layouts reuse the shared Triton kernel.
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
