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

"""MetaX INT8 scaled matrix multiplication with vLLM-compatible arguments.

Public calls consume quantized A/B and explicit scales. Activation quantization
and optional static-weight packing are separate preparation operations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.backends.metax.compiler import MACAOptions
from triton.experimental.tle import is_primitive_supported
from triton.experimental.tle import language as tle_async

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MMA_UNROLL_AVAILABLE = "mma_unroll_count" in MACAOptions.__dataclass_fields__
_SUPPORTED_FLOAT = {torch.bfloat16, torch.float16, torch.float32}
_SMALL_MMA_K16_AVAILABLE = "small_mma_k16" in MACAOptions.__dataclass_fields__
_STREAM_SHARED_MMA_PEEL_AVAILABLE = (
    "stream_shared_mma_peel" in MACAOptions.__dataclass_fields__
)
_STREAM_SHARED_MMA_REVERSE_AVAILABLE = (
    "stream_shared_mma_reverse" in MACAOptions.__dataclass_fields__
)
_STREAM_SHARED_MMA_SMALL_PEEL_AVAILABLE = (
    "stream_shared_mma_small_peel" in MACAOptions.__dataclass_fields__
)
_STREAM_SHARED_MMA_WIDE_PEEL_AVAILABLE = (
    "stream_shared_mma_wide_peel" in MACAOptions.__dataclass_fields__
)
_STREAM_SHARED_MMA_AVAILABLE = "stream_shared_mma" in MACAOptions.__dataclass_fields__
_STREAM_SHARED_MMA_WIDE_AVAILABLE = (
    "stream_shared_mma_tile" in MACAOptions.__dataclass_fields__
)
_STREAM_SHARED_MMA_SMALL_AVAILABLE = (
    "stream_shared_mma_mn" in MACAOptions.__dataclass_fields__
)

_TLE_LOAD_AVAILABLE = is_primitive_supported("metax", "load")


@triton.jit
def _grouped_pids(pid, m, n, block_m, block_n, group_m):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)
    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@libentry()
@triton.jit
def _mm_w8a8_kernel(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    USE_TLE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    PACKED_B: tl.constexpr = False,
    SINGLE_SHARED: tl.constexpr = False,
    SPLIT_ORDER: tl.constexpr = False,
    MASK_M: tl.constexpr = False,
    SPLIT_STORE: tl.constexpr = False,
    SPLIT_GROUP: tl.constexpr = 0,
    K_BEGIN: tl.constexpr = 0,
    K_LENGTH: tl.constexpr = 0,
    STORE_ACC: tl.constexpr = False,
    GROUP_M: tl.constexpr = 8,
    ZERO_PAD_M: tl.constexpr = False,
    PEEL_WIDE_TAIL: tl.constexpr = False,
    TRANSPOSED_OUT: tl.constexpr = False,
    SMALL_MMA_K16: tl.constexpr = False,
    MACHINE_UNROLL: tl.constexpr = 0,
):
    # LibEntry keys constexpr arguments, but not compiler options. Keep the
    # selected layout in that key so aligned and fallback calls cannot alias.
    if SMALL_MMA_K16:
        tl.static_assert(BM == 16 and BN == 64 and BK == 128)
        tl.static_assert(USE_TLE and not SINGLE_SHARED)
    if TRANSPOSED_OUT:
        tl.static_assert(SINGLE_SHARED and not STORE_ACC)
        tl.static_assert(BM == BN and (BM == 64 or BM == 128) and BK == 128)
    if PEEL_WIDE_TAIL:
        tl.static_assert(
            SINGLE_SHARED and USE_TLE and BM == 128 and BN == 128 and BK == 256
        )
    DOMAIN: tl.constexpr = K_LENGTH if K_LENGTH else K
    tl.static_assert(K_BEGIN >= 0 and DOMAIN > 0 and K_BEGIN + DOMAIN <= K)
    if SINGLE_SHARED:
        tl.static_assert(DOMAIN % (SPLIT_K * BK) == 0 and not MASK_M)
        tl.static_assert(
            (BM == 128 and BN == 128 and (BK == 128 or BK == 256))
            or (BM == 64 and BN == 64 and BK == 128)
        )
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    if SPLIT_GROUP:
        tl.static_assert((triton.cdiv(M, BM) * triton.cdiv(N, BN)) % SPLIT_GROUP == 0)
        split_id = (pid // SPLIT_GROUP) % SPLIT_K
        pid = (pid // (SPLIT_GROUP * SPLIT_K)) * SPLIT_GROUP + pid % SPLIT_GROUP
    elif SPLIT_ORDER:
        split_id = pid % SPLIT_K
        pid = pid // SPLIT_K
    pm, pn = _grouped_pids(pid, M, N, BM, BN, GROUP_M)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    CHUNK: tl.constexpr = triton.cdiv(triton.cdiv(DOMAIN, BK), SPLIT_K) * BK
    if SPLIT_K == 1:
        start = K_BEGIN
    else:
        start = K_BEGIN + split_id * CHUNK
    tl.static_assert(SPLIT_K == 1 or CHUNK <= 131071, "split must fit INT32")
    # Quantized operands are contiguous along K; modulo only pads M/N tiles.
    if MASK_M:
        am = rm
    elif ZERO_PAD_M:
        # Only padded output rows repeat row zero; all stored rows use rm.
        am = tl.where(rm < M, rm, 0)
    else:
        am = rm % M
    bn = rn % N
    if M * K > 2147483647:
        am = am.to(tl.int64)
    if N * K > 2147483647:
        bn = bn.to(tl.int64)
    pa = A + am[:, None] * K + start + rk[None, :]
    if PACKED_B:
        if SINGLE_SHARED:
            tl.static_assert(BM == 128 and BN == 128 and BK == 128)
            tl.static_assert(SPLIT_K == 1 and K % BK == 0)
        else:
            tl.static_assert(BN == 256 and BK == 64 and SPLIT_K == 1)
        pb = (
            B
            + (bn[None, :] // BN) * triton.cdiv(K, BK) * BN * BK
            + (bn[None, :] % BN) * BK
            + rk[:, None]
        )
    else:
        pb = B + bn[None, :] * K + start + rk[:, None]
    acc = tl.full((BM, BN), 0, tl.int32)
    if CHUNK > 131071:
        wide = tl.full((BM, BN), 0, tl.int64)
    for i in range(tl.cdiv(CHUNK, BK)):
        if SINGLE_SHARED:
            a = tle_async.load(pa, is_async=True)
            b = tle_async.load(pb, is_async=True)
        elif USE_TLE:
            a = tle_async.load(
                pa,
                (start + i * BK + rk[None, :] < K_BEGIN + DOMAIN)
                & ((rm[:, None] < M) if MASK_M else True),
                other=0,
                is_async=True,
            )
            b = tle_async.load(
                pb,
                start + i * BK + rk[:, None] < K_BEGIN + DOMAIN,
                other=0,
                is_async=True,
            )
        else:
            k_mask = start + i * BK + rk < K_BEGIN + DOMAIN
            a = tl.load(
                pa,
                k_mask[None, :] & ((rm[:, None] < M) if MASK_M else True),
                other=0,
            )
            b = tl.load(pb, k_mask[:, None], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        if CHUNK > 131071:
            # Flush well before signed INT32 overflow, including -128 * -128.
            if (i + 1) % (65536 // BK) == 0:
                wide += acc.to(tl.int64)
                acc = tl.full((BM, BN), 0, tl.int32)
        pa += BK
        pb += BN * BK if PACKED_B else BK
    if CHUNK > 131071:
        value = (wide + acc.to(tl.int64)).to(tl.float32)
    else:
        value = acc.to(tl.float32)
    if SPLIT_K > 1 or STORE_ACC:
        tl.static_assert(CHUNK <= 131071)
        if SPLIT_STORE:
            # Keep the INT32 output layout conversion within 32 KiB so two
            # 128x128 single-buffer CTAs can reside on one C550 SM.
            top, bottom = tl.split(
                tl.permute(tl.reshape(acc, (2, BM // 2, BN)), (1, 2, 0))
            )
            rr = pm * BM + tl.arange(0, BM // 2)
            base = C + split_id.to(tl.int64) * M * N
            tl.store(
                base
                + (
                    rn[None, :].to(tl.int64) * M + rr[:, None]
                    if TRANSPOSED_OUT
                    else rr[:, None].to(tl.int64) * N + rn[None, :]
                ),
                top,
                (rr[:, None] < M) & (rn[None, :] < N),
            )
            rr += BM // 2
            tl.store(
                base
                + (
                    rn[None, :].to(tl.int64) * M + rr[:, None]
                    if TRANSPOSED_OUT
                    else rr[:, None].to(tl.int64) * N + rn[None, :]
                ),
                bottom,
                (rr[:, None] < M) & (rn[None, :] < N),
            )
        else:
            tl.store(
                C
                + split_id.to(tl.int64) * M * N
                + (
                    rn[None, :].to(tl.int64) * M + rm[:, None]
                    if TRANSPOSED_OUT
                    else rm[:, None].to(tl.int64) * N + rn[None, :]
                ),
                acc,
                (rm[:, None] < M) & (rn[None, :] < N),
            )
    else:
        sa = tl.load(SA + rm, rm < M, other=0)
        sb = tl.load(SB + rn, rn < N, other=0)
        if TRANSPOSED_OUT:
            # Preserve the public activation-scale then weight-scale order.
            value = value * sb[None, :] * sa[:, None]
        else:
            value = value * sa[:, None] * sb[None, :]
        if BM == 256 and BN == 256:
            # Convert and store half a tile at a time: converting the full
            # BF16/FP16 tile's MMA layout needs 128 KiB of shared memory.
            value = value.to(C.dtype.element_ty)
            top, bottom = tl.split(
                tl.permute(tl.reshape(value, (2, BM // 2, BN)), (1, 2, 0))
            )
            rr = pm * BM + tl.arange(0, BM // 2)
            tl.store(
                C + rr[:, None].to(tl.int64) * N + rn[None, :],
                top,
                (rr[:, None] < M) & (rn[None, :] < N),
            )
            rr += BM // 2
            tl.store(
                C + rr[:, None].to(tl.int64) * N + rn[None, :],
                bottom,
                (rr[:, None] < M) & (rn[None, :] < N),
            )
        else:
            tl.store(
                C
                + (
                    rn[None, :].to(tl.int64) * M + rm[:, None]
                    if TRANSPOSED_OUT
                    else rm[:, None].to(tl.int64) * N + rn[None, :]
                ),
                value,
                (rm[:, None] < M) & (rn[None, :] < N),
            )


@libentry()
@triton.jit
def _reduce_split_kernel(
    P,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    sk = tl.arange(0, triton.next_power_of_2(SPLITS))
    v = tl.load(
        P + sk[:, None].to(tl.int64) * M * N + x[None, :],
        (sk[:, None] < SPLITS) & (x[None, :] < M * N),
        other=0,
    )
    if K > 131071:
        v = v.to(tl.int64)
    total = tl.sum(v, 0).to(tl.float32)
    sa = tl.load(SA + x // N, x < M * N, other=0)
    sb = tl.load(SB + x % N, x < M * N, other=0)
    tl.store(OUT + x, total * sa * sb, x < M * N)


def _pick_small_mma_k16(a, b, m, n, k, dtype):
    # Two-warp K16 operand layouts reduce register pressure for these measured
    # small-row tiles. Unaligned copies spill heavily, so retain the default
    # layout unless both INT8 operands have the validated alignment.
    return (
        _SMALL_MMA_K16_AVAILABLE
        and dtype in (torch.float16, torch.bfloat16)
        and a.data_ptr() % 128 == 0
        and b.data_ptr() % 128 == 0
        and (
            (m == 1 and n >= 512 and 1024 <= k <= 32768)
            or (2 <= m <= 4 and (n, k) in ((3584, 3584), (4608, 3584)))
            or (m in (2, 4) and (n, k) == (4096, 4096))
            or (m == 8 and (n, k) in ((3584, 3584), (4608, 3584), (14336, 4096)))
        )
    )


# Measured with FlagTune on C550, including the Split-K reduction.
# Values are BN, warps, splits, MASK_M, and the K16 MMA layout selection.
_FLAGTUNE_SMALL_CONFIGS = {
    (1, 6144, 4096): (64, 4, 2, True, False),
    (2, 3584, 3584): (64, 2, 4, False, True),
    (3, 3584, 3584): (32, 2, 4, True, False),
}


def _pick_flagtune_small(a, b, m, n, k, dtype):
    config = _FLAGTUNE_SMALL_CONFIGS.get((m, n, k))
    if (
        config is None
        or dtype not in (torch.float16, torch.bfloat16)
        or a.data_ptr() % 128
        or b.data_ptr() % 128
        or (config[4] and not _SMALL_MMA_K16_AVAILABLE)
    ):
        return None
    return config


def _pick_split(m, n, k):
    if m == 1 and k >= 1024:
        return 8 if n <= 1024 else 4
    if n == 1 and k >= 1024:
        return max(4, triton.next_power_of_2(triton.cdiv(k, 65536)))
    if n == 512 and 1024 <= k <= 4096:
        if 64 <= m <= 128:
            return 4
        if 128 < m <= 256:
            return 2
    if 64 <= m <= 128 and k >= 1024 and (1024 <= n < 2048 or n > 8192 or k > 4096):
        return 8 if n <= 8192 else 1
    if 128 < m <= 256 and n >= 1024 and k >= 1024:
        bm, bn, *_ = _pick_tiles(m, n, k)
        blocks = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        # Split-K helps a very small grid on long K. Shorter K regresses.
        if blocks >= 48 or n >= 4096 or k < 6144:
            return 1
    if m >= 1024 and n >= 1024 and k >= 65536:
        return max(4, triton.next_power_of_2(triton.cdiv(k, 65536)))
    if m <= 256 and n <= 8192 and k >= 1024 and min(m, n) > 1:
        bm, bn, *_ = _pick_tiles(m, n, k)
        blocks = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        if blocks < 104:
            return min(
                16,
                triton.next_power_of_2(
                    triton.cdiv(
                        (
                            104
                            if m <= 16 and k <= 4096 and blocks >= 3 * 104 // 4
                            else 208
                        ),
                        blocks,
                    )
                ),
                max(1, k // 256),
            )
    # Wider M still launches only a few output tiles when N is one tile.
    # Split K until the grid can occupy C550, matching the occupancy rule
    # used by the BF16 MetaX MM path.
    if m > 256 and k >= 2048 and min(m, n) > 1:
        bm, bn, bk, *_ = _pick_tiles(m, n, k)
        blocks = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        if blocks < 104:
            k_iters = max(1, triton.cdiv(k, bk))
            occupancy = triton.next_power_of_2(triton.cdiv(104, blocks))
            max_splits = max(1, k_iters // 2)
            return max(1, min(32, occupancy, max_splits))
    return 1


def _pick_tiles(m, n, k):
    # On C550's 104 SMs, this range fits one wave of 256x256 tiles
    # while 128x128 tiles need at least three waves.
    if m == 256 and 104 * 128 < n <= 104 * 256 and 1024 <= k <= 4096:
        return (256, 256, 64, 8, 2, "basic", _TLE_LOAD_AVAILABLE)
    if 128 < m <= 256 and n == 512 and 1024 <= k <= 4096 and k % 128 == 0:
        return (64, 64, 128, 4, 2, "cpasync-mixed", _TLE_LOAD_AVAILABLE)
    if (m >= 4096 and n >= 2048 and 1024 <= k <= 32768) or (
        m == 256 and n >= 65536 and 1024 <= k <= 4096
    ):
        return (256, 256, 64, 8, 2, "basic", _TLE_LOAD_AVAILABLE)
    if 1024 <= k <= 4096 and ((m >= 4096 and n >= 1024) or (m == 256 and n >= 32768)):
        return (256, 128, 64, 8, 2, "basic", _TLE_LOAD_AVAILABLE)
    if m <= 16:
        # Wide outputs already provide enough CTAs; fewer warps reduce
        # per-CTA resource usage without adding split-K work.
        warps = 2 if n > 8192 and 1024 <= k <= 4096 else 4
        return (
            16,
            64,
            128 if k >= 1024 else 64,
            warps,
            2,
            "basic",
            _TLE_LOAD_AVAILABLE,
        )
    if m >= 1024 and n >= 1024:
        if k >= 65536:
            return (128, 128, 256, 8, 1, "basic", _TLE_LOAD_AVAILABLE)
        return (128, 128, 128, 8, 2, "basic", _TLE_LOAD_AVAILABLE)
    if (
        ((64 <= m <= 128 and (1024 <= n < 2048 or n > 8192 or k > 4096)) or m >= 1024)
        and n >= 512
        and k >= 1024
    ):
        return (128, 128, 128, 8, 2, "basic", _TLE_LOAD_AVAILABLE)
    if 128 < m <= 256 and n >= 2048 and k >= 1024:
        return (128, 128, 128, 8, 2, "basic", _TLE_LOAD_AVAILABLE)
    if m >= 128 and k <= 512:
        return (32, 64, 64, 4, 2, "basic", _TLE_LOAD_AVAILABLE)
    return (
        32 if m < 64 else 64,
        64,
        128 if k >= 1024 else 64,
        4,
        2,
        "basic",
        _TLE_LOAD_AVAILABLE,
    )


@libentry()
@triton.jit
def _zero_output_kernel(OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + offsets, 0, offsets < SIZE)


def _validate_mm_inputs(a, b, scale_a, scale_b):
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_a, scale_b)):
        raise TypeError("mm_w8a8_int8 expects Tensor inputs and scales")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("A and B must be prequantized torch.int8 tensors")
    if a.device.type != "cuda" or any(
        x.device != a.device for x in (b, scale_a, scale_b)
    ):
        raise ValueError("inputs and scales must be on the same MetaX device")
    m, k = a.shape
    n = b.shape[1]
    if scale_a.shape not in ((m,), (m, 1)) or scale_b.shape not in ((n,), (1, n)):
        raise ValueError("expected scale_a[M] or [M,1], scale_b[N] or [1,N]")
    if any(
        x.dtype != torch.float32 or not x.is_contiguous() for x in (scale_a, scale_b)
    ):
        raise ValueError("scales must be contiguous FP32 tensors")
    return m, n, k


def _pick_stream_shared(m, n, k, dtype):
    # Enable the guarded streaming schedule only for measured configurations.
    if not _STREAM_SHARED_MMA_AVAILABLE or dtype not in (torch.bfloat16, torch.float16):
        return None
    if m == 256:
        if (n, k) in ((14336, 4096), (18944, 3584), (28672, 4096)):
            return (1, 1)
        if (n, k) in ((4096, 14336), (3584, 18944)):
            return (4, 1)
    if (m, n, k) == (2048, 2048, 2048):
        return (1, 1)
    if (m, n, k) == (8192, 1024, 4096):
        return (1, 2)
    if 64 <= m <= 128:
        if (n, k) == (3584, 18944):
            return (4, 1)
        if (n, k) == (4096, 14336):
            return (8, 2)
    if not 64 <= m <= 128 or (n, k) not in (
        (14336, 4096),
        (16384, 4096),
        (18944, 3584),
        (28672, 4096),
    ):
        return None
    return (2, 1) if m < 128 and n <= 16384 else (1, 2)


def _pick_aligned_stream(a, b, m, n, k, dtype):
    # The square 4096 GEMM uses the measured four-warp streaming tile.
    if (
        _STREAM_SHARED_MMA_AVAILABLE
        and _STREAM_SHARED_MMA_REVERSE_AVAILABLE
        and _STREAM_SHARED_MMA_PEEL_AVAILABLE
        and (m, n, k) == (4096, 4096, 4096)
        and dtype in (torch.float16, torch.bfloat16)
        and a.data_ptr() % 128 == 0
        and b.data_ptr() % 128 == 0
    ):
        return (1, 1)
    if (
        not _STREAM_SHARED_MMA_AVAILABLE
        or dtype not in (torch.float16, torch.bfloat16)
        or (m, n, k) != (256, 37888, 3584)
        or a.data_ptr() % 16
        or b.data_ptr() % 16
    ):
        return None
    return (1, 1)


def _pick_small_stream(a, b, m, n, k, dtype):
    if not _STREAM_SHARED_MMA_SMALL_AVAILABLE or dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return None
    # The measured direct-copy path benefits from aligned 128-byte segments.
    if a.data_ptr() % 128 or b.data_ptr() % 128:
        return None
    if 64 <= m <= 128 and (n, k) == (512, 3584):
        return (7, 1)
    if (64 <= m <= 128 or m == 256) and (n, k) == (1024, 4096):
        return (4, 1)
    if (m, n, k) == (256, 512, 3584):
        return (4, 8)
    if 64 <= m <= 128 and (n, k) in ((3584, 3584), (4096, 4096)):
        return (2, 1)
    if m == 256 and (n, k) in ((3584, 3584), (4096, 4096), (4608, 3584)):
        return (1, 1 if n == 3584 else 8)
    return None


def _launch_wide_stream(a, sa, b, sb, out, m, n, k):
    # Preserve original row strides. Process full K tiles, then any remainder,
    # and combine all integer partials before scaling exactly once.
    peel_wide = (
        _STREAM_SHARED_MMA_WIDE_PEEL_AVAILABLE
        and a.data_ptr() % 128 == 0
        and b.data_ptr() % 128 == 0
    )
    splits = 4 if k == 128256 else 8
    prefix = (k // (splits * 256)) * (splits * 256)
    tail = k - prefix
    parts = splits + int(tail > 0)
    partial = torch.empty((parts, m, n), device=out.device, dtype=torch.int32)
    blocks = triton.cdiv(m, 128) * triton.cdiv(n, 128)
    with torch_device_fn.device(a.device):
        _mm_w8a8_kernel[(blocks, splits)](
            a,
            b,
            sa,
            sb,
            partial,
            m,
            n,
            k,
            128,
            128,
            256,
            True,
            splits,
            SINGLE_SHARED=True,
            SPLIT_STORE=True,
            K_LENGTH=prefix,
            num_warps=8,
            num_stages=2,
            pipeline="basic",
            pipeline_load_num=2,
            single_shared_pipeline=True,
            single_shared_async=True,
            stream_shared_mma=1,
            stream_shared_mma_tile=256,
            PEEL_WIDE_TAIL=peel_wide,
            enable_fp_fusion=False,
            **({"stream_shared_mma_wide_peel": True} if peel_wide else {}),
        )
        if tail:
            _mm_w8a8_kernel[(blocks,)](
                a,
                b,
                sa,
                sb,
                partial[splits],
                m,
                n,
                k,
                128,
                128,
                128,
                True,
                1,
                STORE_ACC=True,
                SPLIT_STORE=True,
                K_BEGIN=prefix,
                K_LENGTH=tail,
                num_warps=8,
                num_stages=2,
                pipeline="basic",
                enable_fp_fusion=False,
            )
        _reduce_split_kernel[(triton.cdiv(m * n, 1024),)](
            partial,
            sa,
            sb,
            out,
            m,
            n,
            k,
            parts,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def _pick_stream_split_group(m, n, k):
    # Group adjacent output tiles before advancing the K partition.
    if (m, n, k) == (98, 14336, 4096):
        return 56
    if (m, n, k) == (98, 4096, 14336):
        return 16
    return 0


def _pick_zero_pad_m(a, b, m, n, k, dtype):
    return (
        _STREAM_SHARED_MMA_AVAILABLE
        and dtype in (torch.float16, torch.bfloat16)
        and 64 <= m < 128
        and (n, k) in ((37888, 3584), (128256, 4096), (152064, 3584))
        and a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
    )


def _pick_transposed_stream(a, b, m, n, k, dtype):
    return (
        _STREAM_SHARED_MMA_REVERSE_AVAILABLE
        and _STREAM_SHARED_MMA_PEEL_AVAILABLE
        and dtype in (torch.float16, torch.bfloat16)
        and (
            (m == 8192 and (n, k) in ((512, 3584), (1024, 4096)))
            or (
                64 <= m <= 128
                and (n, k)
                in (
                    (28672, 4096),
                    (37888, 3584),
                    (128256, 4096),
                    (152064, 3584),
                )
            )
        )
        and a.data_ptr() % 128 == 0
        and b.data_ptr() % 128 == 0
    )


def _launch(a_q, a_scale, b_q, b_scale, out, m, n, k):
    # Two K partitions retain useful parallelism for the measured M=98 tiles.
    # Partial sums use the original output orientation before scale reduction.
    if (
        _STREAM_SHARED_MMA_SMALL_PEEL_AVAILABLE
        and m == 98
        and (n, k) in ((4096, 4096), (4608, 3584))
        and out.dtype in (torch.float16, torch.bfloat16)
        and a_q.data_ptr() % 128 == 0
        and b_q.data_ptr() % 128 == 0
    ):
        target = torch.empty((2, m, n), device=out.device, dtype=torch.int32)
        # FlagTune confirmed adjacent K partitions for this transposed tile.
        tuned_grid = (n, k) == (4608, 3584)
        blocks = triton.cdiv(n, 64) * triton.cdiv(m, 64)
        grid = (blocks * 2,) if tuned_grid else (blocks, 2)
        with torch_device_fn.device(a_q.device):
            _mm_w8a8_kernel[grid](
                b_q,
                a_q,
                b_scale,
                a_scale,
                target,
                n,
                m,
                k,
                64,
                64,
                128,
                True,
                2,
                SINGLE_SHARED=True,
                SPLIT_STORE=True,
                TRANSPOSED_OUT=True,
                SPLIT_GROUP=1 if tuned_grid else 0,
                GROUP_M=16 if tuned_grid else 8,
                num_warps=4,
                num_stages=2,
                pipeline="basic",
                scenario="unprefetch",
                enable_fp_fusion=False,
                pipeline_load_num=2,
                single_shared_pipeline=True,
                single_shared_async=True,
                stream_shared_mma=1,
                stream_shared_mma_mn=64,
                stream_shared_mma_small_peel=True,
            )
            _reduce_split_kernel[(triton.cdiv(m * n, 1024),)](
                target,
                a_scale,
                b_scale,
                out,
                m,
                n,
                k,
                2,
                1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out
    if _pick_transposed_stream(a_q, b_q, m, n, k, out.dtype):
        # Swap the logical GEMM axes and write directly to the public layout.
        with torch_device_fn.device(a_q.device):
            _mm_w8a8_kernel[(triton.cdiv(n, 128) * triton.cdiv(m, 128),)](
                b_q,
                a_q,
                b_scale,
                a_scale,
                out,
                n,
                m,
                k,
                128,
                128,
                128,
                True,
                1,
                SINGLE_SHARED=True,
                TRANSPOSED_OUT=True,
                GROUP_M=1 if m <= 128 else 8,
                num_warps=4,
                num_stages=2,
                pipeline="basic",
                pipeline_load_num=2,
                single_shared_pipeline=True,
                single_shared_async=True,
                stream_shared_mma=1,
                stream_shared_mma_reverse=True,
                stream_shared_mma_peel=True,
                enable_fp_fusion=False,
            )
        return out
    if (
        _STREAM_SHARED_MMA_WIDE_AVAILABLE
        and out.dtype in (torch.float16, torch.bfloat16)
        and m == 1848
        and n == 1536
        and k in (128256, 151936, 152064)
    ):
        return _launch_wide_stream(a_q, a_scale, b_q, b_scale, out, m, n, k)
    bm, bn, bk, warps, stages, pipeline, use_tle = _pick_tiles(m, n, k)
    # FP32 output requires more layout-conversion storage per element.
    if bm == 256 and out.dtype == torch.float32:
        bm, bn, bk = 128, 128, 128
    splits = _pick_split(m, n, k)
    stream = _pick_aligned_stream(a_q, b_q, m, n, k, out.dtype) or _pick_stream_shared(
        m, n, k, out.dtype
    )
    # The guarded streaming pass requires vectorized 16-byte copies.
    # Unaligned inputs use the existing general TLE pipeline.
    if a_q.data_ptr() % 16 or b_q.data_ptr() % 16:
        stream = None
    small_stream = _pick_small_stream(a_q, b_q, m, n, k, out.dtype)
    if small_stream is not None:
        stream = (small_stream[0], 1)
    # The source-built MetaX compiler has a stub cpasync-mixed pass.
    # Use the measured basic TLE pipeline for this affected configuration.
    if _STREAM_SHARED_MMA_AVAILABLE and (m, n, k) == (256, 512, 3584):
        bm, bn, bk, warps, stages, pipeline, use_tle = (
            64,
            64,
            128,
            4,
            2,
            "basic",
            True,
        )
        splits = 4
    stream_options = {}
    if (
        2 <= m <= 16
        and (n, k) == (4096, 14336)
        and out.dtype in (torch.float16, torch.bfloat16)
    ):
        stream_options["scenario"] = "unprefetch"
    if stream is not None:
        splits, chains = stream
        bm, bn, bk, warps, stages, pipeline, use_tle = (
            128,
            128,
            128,
            4,
            2,
            "basic",
            True,
        )
        stream_options = dict(
            pipeline_load_num=2,
            single_shared_pipeline=True,
            single_shared_async=True,
            stream_shared_mma=chains,
        )
        if (
            _STREAM_SHARED_MMA_REVERSE_AVAILABLE
            and small_stream is None
            and out.dtype in (torch.float16, torch.bfloat16)
            and a_q.data_ptr() % 16 == 0
            and b_q.data_ptr() % 16 == 0
        ):
            stream_options["stream_shared_mma_reverse"] = True
            if _STREAM_SHARED_MMA_PEEL_AVAILABLE and (m, n, k) != (8192, 1024, 4096):
                stream_options["stream_shared_mma_peel"] = True
        if small_stream is not None:
            bm = bn = 64
            stream_options.update(stream_shared_mma_mn=64, scenario="unprefetch")
            if _STREAM_SHARED_MMA_SMALL_PEEL_AVAILABLE:
                stream_options["stream_shared_mma_small_peel"] = True
    small_mma_k16 = stream is None and _pick_small_mma_k16(a_q, b_q, m, n, k, out.dtype)
    if small_mma_k16:
        warps = 2
        stream_options["small_mma_k16"] = True
    tuned_small = _pick_flagtune_small(a_q, b_q, m, n, k, out.dtype)
    if tuned_small is not None:
        bn, warps, splits, tuned_mask_m, small_mma_k16 = tuned_small
        bm, bk = 16, 128
        if small_mma_k16:
            stream_options["small_mma_k16"] = True
        else:
            stream_options.pop("small_mma_k16", None)
    # The tuned shapes differ from their general variants in BN, SPLIT_K,
    # MASK_M, or SMALL_MMA_K16, which also distinguish the LibEntry cache keys.
    # Keep machine unrolling separate from the two-stage frontend pipeline.
    # The constexpr below also distinguishes aligned and fallback LibEntry keys.
    machine_unroll = 0
    if (
        _MMA_UNROLL_AVAILABLE
        and (m, n, k) in ((1024, 1024, 1024), (256, 6144, 4096))
        and out.dtype in (torch.float16, torch.bfloat16)
        and a_q.data_ptr() % 128 == 0
        and b_q.data_ptr() % 128 == 0
    ):
        machine_unroll = 8
        stream_options["mma_unroll_count"] = machine_unroll
    if splits > 1:
        # Every integer partial must stay below the signed INT32 limit.
        splits = max(splits, triton.next_power_of_2(triton.cdiv(k, 65536)))
    target = (
        out
        if splits == 1
        else torch.empty((splits, m, n), device=out.device, dtype=torch.int32)
    )
    # Adjacent K partitions reduce the measured latency variation for this
    # small-row, power-of-two weight matrix on C550. Other shapes retain the
    # original grid because the same order can regress them.
    split_order = m == 8 and n == 4096 and k == 4096
    blocks = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    split_group = 16 if (m, n, k) == (3, 4096, 4096) else 0
    mask_m = split_order or bool(split_group)
    if (
        (2 <= m <= 6 and (n, k) == (3584, 3584))
        or (3 <= m <= 8 and (n, k) == (6144, 4096))
    ) and out.dtype in (torch.float16, torch.bfloat16):
        mask_m = True
    if tuned_small is not None:
        mask_m = tuned_mask_m
    if stream is not None:
        split_group = _pick_stream_split_group(m, n, k) or split_group
    grid = (blocks * splits,) if split_order or split_group else (blocks, splits)
    with torch_device_fn.device(a_q.device):
        _mm_w8a8_kernel[grid](
            a_q,
            b_q,
            a_scale,
            b_scale,
            target,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            use_tle,
            splits,
            SPLIT_ORDER=split_order,
            MASK_M=mask_m,
            ZERO_PAD_M=_pick_zero_pad_m(a_q, b_q, m, n, k, out.dtype),
            SPLIT_GROUP=split_group,
            GROUP_M=(
                4
                if (m, n, k) == (8192, 3584, 18944)
                and out.dtype == torch.bfloat16
                and a_q.data_ptr() % 128 == 0
                and b_q.data_ptr() % 128 == 0
                else (
                    small_stream[1]
                    if small_stream is not None
                    else (
                        1
                        if 64 < m <= 128
                        and (n, k)
                        in ((3584, 3584), (4096, 4096), (4608, 3584), (6144, 4096))
                        else 8
                    )
                )
            ),
            SINGLE_SHARED=stream is not None,
            SMALL_MMA_K16=small_mma_k16,
            MACHINE_UNROLL=machine_unroll,
            SPLIT_STORE=stream is not None and splits > 1,
            num_warps=warps,
            num_stages=stages,
            pipeline=pipeline,
            enable_fp_fusion=False,
            **stream_options,
        )
        if splits > 1:
            reduce_block = 512 if m <= 16 and n >= 2048 else 256
            reduce_warps = (2 if n >= 2048 else 1) if m <= 16 else 4
            # Larger output ranges need several values per thread; 256-element
            # blocks otherwise launch many short reduction CTAs on C550.
            if m >= 64 and m * n >= 96 * 1024:
                reduce_block, reduce_warps = 1024, 4
            # The measured 512-element/two-warp layout avoids shared-memory
            # reduction for these four- and seven-partition small-N cases.
            if 64 <= m <= 128 and n == 512 and splits in (4, 7):
                reduce_block, reduce_warps = 512, 2
            if splits > 32:
                reduce_block = min(256, max(1, 8192 // triton.next_power_of_2(splits)))
                reduce_warps = 4
            _reduce_split_kernel[(triton.cdiv(m * n, reduce_block),)](
                target,
                a_scale,
                b_scale,
                out,
                m,
                n,
                k,
                splits,
                reduce_block,
                num_warps=reduce_warps,
                enable_fp_fusion=False,
            )
    return out


@libentry()
@triton.jit
def _mm_w8a8_tle_vector_kernel(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
):
    tl.static_assert(K > 0 and K % 16 == 0 and N > 0)
    tl.static_assert(BK >= 128, "vector TLE requires at least 128 columns")
    row = tl.program_id(1)
    ks = tl.arange(0, BK)
    cols = tl.program_id(0) * BR + tl.arange(0, BR)
    a = tle_async.load(A + row * K + (ks % K)[None, :], is_async=True).to(tl.int32)
    b = tle_async.load(
        B + (cols % N)[:, None] * K + (ks % K)[None, :], is_async=True
    ).to(tl.int32)
    value = tl.sum(tl.where(ks[None, :] < K, a * b, 0), 1).to(tl.float32)
    sa = tl.load(SA + row)
    sb = tl.load(SB + cols, cols < N, other=0)
    tl.store(C + row * N + cols, value * sa * sb, cols < N)


def _run_mm(a, b, scale_a, scale_b, out, m, n, k):
    if m == 0 or n == 0:
        return out
    if k == 0:
        with torch_device_fn.device(a.device):
            _zero_output_kernel[(triton.cdiv(m * n, 1024),)](out, m * n, BLOCK=1024)
        return out
    a = a.contiguous()
    b = b.t().contiguous().t()
    if (
        m == 1
        and n <= 1024
        and 128 <= k <= 4096
        and k % 16 == 0
        and "vector_tle" in MACAOptions.__dataclass_fields__
    ):
        with torch_device_fn.device(a.device):
            _mm_w8a8_tle_vector_kernel[(triton.cdiv(n, 2), m)](
                a,
                b,
                scale_a,
                scale_b,
                out,
                m,
                n,
                k,
                triton.next_power_of_2(k),
                2,
                num_warps=4,
                num_stages=2,
                pipeline="basic",
                vector_tle=True,
                enable_fp_fusion=False,
            )
        return out
    if (m, n, k) == (2, 512, 3584) and _SMALL_MMA_K16_AVAILABLE:
        with torch_device_fn.device(a.device):
            partial = torch.empty((8, m, n), device=a.device, dtype=torch.int32)
            _mm_w8a8_kernel[(8, 8)](
                a,
                b,
                scale_a,
                scale_b,
                partial,
                m,
                n,
                k,
                16,
                64,
                128,
                True,
                8,
                SMALL_MMA_K16=True,
                num_warps=2,
                num_stages=2,
                pipeline="basic",
                small_mma_k16=True,
                enable_fp_fusion=False,
            )
            _reduce_split_kernel[(2,)](
                partial,
                scale_a,
                scale_b,
                out,
                m,
                n,
                k,
                8,
                512,
                num_warps=2,
                enable_fp_fusion=False,
            )
        return out
    return _launch(a, scale_a, b, scale_b, out, m, n, k)


def _mm_w8a8_int8_prequantized(a, b, scale_a, scale_b, *, out_dtype=torch.bfloat16):
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if out_dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out_dtype must be BF16, FP16 or FP32")
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k)


def _mm_w8a8_int8_prequantized_out(a, b, scale_a, scale_b, *, out):
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    if out.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out must be BF16, FP16 or FP32")
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k)


@libentry()
@triton.jit
def _quantize_mm_input_kernel(
    X,
    PEAK,
    Q,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    STRIDE_R: tl.constexpr,
    STRIDE_C: tl.constexpr,
    PER_ROW: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if COLUMN_MAJOR:
        row, col = offsets % ROWS, offsets // ROWS
    else:
        row, col = offsets // COLS, offsets % COLS
    mask = offsets < ROWS * COLS
    value = tl.load(X + row * STRIDE_R + col * STRIDE_C, mask, other=0).to(tl.float32)
    peak = tl.load(PEAK + (row if PER_ROW else col), mask, other=1)
    # Correctly-rounded division avoids platform-dependent reciprocal error
    # changing the integer code at half-integer quantization boundaries.
    normalized = tl.div_rn(value, peak) * 127.0
    lower = tl.floor(normalized)
    frac = normalized - lower
    odd = (lower.to(tl.int32) & 1) != 0
    rounded = lower + tl.where((frac > 0.5) | ((frac == 0.5) & odd), 1.0, 0.0)
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(Q + offsets, quantized, mask)


@libentry()
@triton.jit
def _quantize_rows_kernel(
    X,
    Q,
    SCALE,
    R: tl.constexpr,
    K: tl.constexpr,
    SR: tl.constexpr,
    SK: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    ks = tl.arange(0, BK)
    x = tl.load(
        X + rows[:, None].to(tl.int64) * SR + ks[None, :].to(tl.int64) * SK,
        (rows[:, None] < R) & (ks[None, :] < K),
        other=0,
    ).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-10)
    value = tl.div_rn(x, peak[:, None]) * 127.0
    lower = tl.floor(value)
    frac = value - lower
    odd = (lower.to(tl.int32) & 1) != 0
    rounded = lower + tl.where((frac > 0.5) | ((frac == 0.5) & odd), 1.0, 0.0)
    code = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(
        Q + rows[:, None].to(tl.int64) * K + ks[None, :],
        code,
        (rows[:, None] < R) & (ks[None, :] < K),
    )
    tl.store(SCALE + rows, peak * (1.0 / 127.0), rows < R)


def _prepare_mm_w8a8_int8_inputs(a, b):
    """Quantize current inputs directly into row-major A and column-major B."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("mm_w8a8_int8 expects Tensor inputs")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in _SUPPORTED_FLOAT or b.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("A and B must be FP16, BF16 or FP32 tensors")
    if a.device.type != "cuda" or a.device != b.device:
        raise ValueError("A and B must be on the same MetaX device")
    m, k = a.shape
    n = b.shape[1]
    # Empty outputs and empty reductions must not call amax on an empty axis.
    if m == 0 or n == 0 or k == 0:
        return (
            torch.empty((m, k), device=a.device, dtype=torch.int8),
            torch.empty((k, n), device=a.device, dtype=torch.int8),
            torch.ones(m, device=a.device, dtype=torch.float32),
            torch.ones(n, device=a.device, dtype=torch.float32),
        )
    scale_a = torch.empty(m, device=a.device, dtype=torch.float32)
    scale_b = torch.empty(n, device=a.device, dtype=torch.float32)
    a_q = torch.empty((m, k), device=a.device, dtype=torch.int8)
    b_q = torch.empty_strided((k, n), (1, k), device=a.device, dtype=torch.int8)
    with torch_device_fn.device(a.device):
        for x, q, scale in ((a, a_q, scale_a), (b.t(), b_q.t(), scale_b)):
            if k >= 1024 and x.stride(1) != 1:
                peak = x.float().abs().amax(dim=1).clamp_min(1e-10)
                torch.mul(peak, 1.0 / 127.0, out=scale)
                physical_q = torch.empty_strided(
                    x.shape, (1, x.shape[0]), device=x.device, dtype=torch.int8
                )
                _quantize_mm_input_kernel[(triton.cdiv(x.numel(), 1024),)](
                    x,
                    peak,
                    physical_q,
                    *x.shape,
                    *x.stride(),
                    True,
                    True,
                    BLOCK=1024,
                    enable_fp_fusion=False,
                )
                q.copy_(physical_q)
            elif k <= 16384:
                br = 1 if x.stride(1) == 1 else 16
                bk = triton.next_power_of_2(k)
                br = min(br, max(1, 16384 // bk))
                _quantize_rows_kernel[(triton.cdiv(x.shape[0], br),)](
                    x,
                    q,
                    scale,
                    x.shape[0],
                    k,
                    *x.stride(),
                    br,
                    bk,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                # Bound per-program reduction storage for very large K.
                peak = x.float().abs().amax(dim=1).clamp_min(1e-10)
                torch.mul(peak, 1.0 / 127.0, out=scale)
                _quantize_mm_input_kernel[(triton.cdiv(x.numel(), 1024),)](
                    x,
                    peak,
                    q,
                    *x.shape,
                    *x.stride(),
                    True,
                    False,
                    BLOCK=1024,
                    enable_fp_fusion=False,
                )
    return a_q, b_q, scale_a, scale_b


def _mm_w8a8_int8_floating(a, b, *, out_dtype=None):
    """Compute W8A8 GEMM from floating inputs using symmetric INT8 quantization.

    The call signature follows mm_w8a8_fp8; the quantization format is INT8.
    Default output is BF16. Quantization is recomputed, never cached by pointer.
    """
    logger.debug("GEMS MM_W8A8_INT8")
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out_dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out_dtype must be BF16, FP16 or FP32")
    return _mm_w8a8_int8_prequantized(
        *_prepare_mm_w8a8_int8_inputs(a, b), out_dtype=out_dtype
    )


def _mm_w8a8_int8_floating_out(a, b, *, out):
    """Write floating-input INT8 GEMM into a contiguous caller-owned output."""
    logger.debug("GEMS MM_W8A8_INT8_OUT")
    return _mm_w8a8_int8_prequantized_out(*_prepare_mm_w8a8_int8_inputs(a, b), out=out)


@triton.jit
def _activation_code(x, peak, LOW_PRECISION: tl.constexpr):
    if LOW_PRECISION:
        fast = (peak > 1.0e-10) & (peak < 1.0e30)
    else:
        fast = False
    if fast:
        normalized = x * tl.div_rn(127.0, peak)
        # In this range, adding 1.5 * 2**23 rounds FP32 to an even integer.
        rounded = (normalized + 12582912.0) - 12582912.0
        # BF16/FP16 products below are exact in FP32. Correct reciprocal
        # error at exact half-integer boundaries without per-element div.
        distance = x * 254.0 - peak * (rounded * 2.0)
        odd = (rounded.to(tl.int32) & 1) != 0
        rounded = tl.where(
            odd & (tl.abs(distance) == peak),
            rounded + tl.where(distance > 0, 1.0, -1.0),
            rounded,
        )
    else:
        # Preserve the original division and rounding for FP32 and extreme peaks.
        normalized = tl.div_rn(x, peak) * 127.0
        lower = tl.floor(normalized)
        frac = normalized - lower
        odd = (lower.to(tl.int32) & 1) != 0
        rounded = lower + tl.where((frac > 0.5) | ((frac == 0.5) & odd), 1.0, 0.0)
        rounded = tl.minimum(tl.maximum(rounded, -127.0), 127.0)

    return rounded.to(tl.int8)


@libentry()
@triton.jit
def _activation_rows_kernel(X, Q, SCALE, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    ks = tl.arange(0, BK)
    x = tl.load(X + row.to(tl.int64) * K + ks, ks < K, other=0).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-10)
    code = _activation_code(
        x, peak, X.dtype.element_ty == tl.bfloat16 or X.dtype.element_ty == tl.float16
    )
    tl.store(Q + row.to(tl.int64) * K + ks, code, ks < K)
    tl.store(SCALE + row, peak * (1.0 / 127.0))


@libentry()
@triton.jit
def _activation_peaks_kernel(
    X, PEAKS, K: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    ks = chunk * BK + tl.arange(0, BK)
    x = tl.load(X + row.to(tl.int64) * K + ks, ks < K, other=0).to(tl.float32)
    tl.store(PEAKS + row.to(tl.int64) * SPLITS + chunk, tl.max(tl.abs(x), 0))


@libentry()
@triton.jit
def _activation_chunks_kernel(
    X, PEAKS, Q, SCALE, K: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    sp = tl.arange(0, triton.next_power_of_2(SPLITS))
    peak = tl.maximum(
        tl.max(
            tl.load(PEAKS + row.to(tl.int64) * SPLITS + sp, sp < SPLITS, other=0), 0
        ),
        1.0e-10,
    )
    ks = chunk * BK + tl.arange(0, BK)
    x = tl.load(X + row.to(tl.int64) * K + ks, ks < K, other=0).to(tl.float32)
    code = _activation_code(
        x, peak, X.dtype.element_ty == tl.bfloat16 or X.dtype.element_ty == tl.float16
    )
    tl.store(Q + row.to(tl.int64) * K + ks, code, ks < K)
    if chunk == 0:
        tl.store(SCALE + row, peak * (1.0 / 127.0))


def _prepare_mm_w8a8_int8_activation(a):
    """Quantize current activations when INT8 weights are already available."""
    if a.ndim != 2 or a.dtype not in _SUPPORTED_FLOAT or a.device.type != "cuda":
        raise ValueError("expected a floating MetaX activation matrix")
    a = a.contiguous()
    m, k = a.shape
    q = torch.empty_like(a, dtype=torch.int8)
    scale = torch.empty(m, device=a.device, dtype=torch.float32)
    if m == 0 or k == 0:
        scale.fill_(1)
        return q, scale
    with torch_device_fn.device(a.device):
        # Bound per-thread live values to avoid spilling a 16K row.
        if k <= 8192:
            _activation_rows_kernel[(m,)](
                a,
                q,
                scale,
                k,
                triton.next_power_of_2(k),
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif k <= 1048576:
            bk = 4096
            splits = triton.cdiv(k, bk)
            peaks = torch.empty((m, splits), device=a.device, dtype=torch.float32)
            _activation_peaks_kernel[(m, splits)](
                a,
                peaks,
                k,
                bk,
                splits,
                num_warps=4,
            )
            _activation_chunks_kernel[(m, splits)](
                a,
                peaks,
                q,
                scale,
                k,
                bk,
                splits,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            # Bound the partial-peak reduction size for unusually long rows.
            peak = a.float().abs().amax(dim=1).clamp_min(1e-10)
            torch.mul(peak, 1.0 / 127.0, out=scale)
            _quantize_mm_input_kernel[(triton.cdiv(a.numel(), 1024),)](
                a,
                peak,
                q,
                m,
                k,
                k,
                1,
                True,
                False,
                BLOCK=1024,
                enable_fp_fusion=False,
            )
    return q, scale


def _mm_w8a8_int8_prepared_weight_out(a, bq, scale_b, *, out):
    """Prepare dynamic activations, then execute the TLE matrix multiply."""
    aq, scale_a = _prepare_mm_w8a8_int8_activation(a)
    return _mm_w8a8_int8_prequantized_out(aq, bq, scale_a, scale_b, out=out)


@dataclass(frozen=True)
class _PackedMMW8A8Weight:
    """Explicit snapshot of static INT8 weights; retains a standard fallback."""

    standard: torch.Tensor
    scale: torch.Tensor
    tiled: torch.Tensor | None
    single_shared: bool = False


def _pack_mm_w8a8_int8_weight(bq, scale_b, *, single_shared=False):
    """Prepare static weights once, outside repeated activation/GEMM calls.

    The tiled path uses an additional INT8 copy. Preparing a new object is
    required when weights or scales change; no input content is cached.
    single_shared opts into a 128x128 weight layout and the experimental
    FlagTree direct-copy pipeline. It is not enabled by default.
    """
    if type(single_shared) is not bool:
        raise TypeError("single_shared must be bool")
    if single_shared:
        from triton.backends.metax.compiler import MACAOptions

        if "single_shared_async" not in MACAOptions.__dataclass_fields__:
            raise RuntimeError(
                "single_shared requires a FlagTree build with single_shared_async"
            )
    if not isinstance(bq, torch.Tensor) or not isinstance(scale_b, torch.Tensor):
        raise TypeError("weight and scale must be tensors")
    if bq.ndim != 2 or bq.dtype != torch.int8 or bq.device.type != "cuda":
        raise ValueError("expected MetaX INT8 weight [K,N]")
    k, n = bq.shape
    if (
        scale_b.device != bq.device
        or scale_b.dtype != torch.float32
        or scale_b.shape not in ((n,), (1, n))
        or not scale_b.is_contiguous()
    ):
        raise ValueError("expected contiguous FP32 weight scale [N] or [1,N]")
    standard = bq.t().clone(memory_format=torch.contiguous_format).t()
    scale = scale_b.reshape(-1).clone()
    tiled = None
    bn, bk = (128, 128) if single_shared else (256, 64)
    # This N/K pair did not show a stable packing benefit in paired runs.
    if (
        n >= 2048
        and 1024 <= k <= 32768
        and n % bn == 0
        and k % bk == 0
        and (n, k) != (4608, 3584)
    ):
        tiled = (
            standard.t()
            .reshape(n // bn, bn, k // bk, bk)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
    return _PackedMMW8A8Weight(standard, scale, tiled, single_shared)


def _mm_w8a8_int8_packed_prequantized_out(aq, scale_a, weight, *, out):
    """Use prequantized activations with an explicitly prepared weight snapshot."""
    if not isinstance(weight, _PackedMMW8A8Weight):
        raise TypeError("weight must be prepared by _pack_mm_w8a8_int8_weight")
    m, n, k = _validate_mm_inputs(aq, weight.standard, scale_a, weight.scale)
    if out.shape != (m, n) or out.device != aq.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    if out.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out must be BF16, FP16 or FP32")
    use_tiled = (
        weight.tiled is not None
        and m >= 4096
        and aq.is_contiguous()
        and out.dtype == torch.bfloat16
        and _pick_tiles(m, n, k)[:3] == (256, 256, 64)
    )
    if not use_tiled:
        return _run_mm(aq, weight.standard, scale_a, weight.scale, out, m, n, k)
    if (
        weight.tiled.device != aq.device
        or weight.tiled.dtype != torch.int8
        or not weight.tiled.is_contiguous()
        or weight.tiled.numel() != n * k
    ):
        raise ValueError("invalid tiled INT8 weight storage")
    if weight.single_shared:
        if weight.tiled.shape != (n // 128, k // 128, 128, 128) or k % 128:
            raise ValueError("invalid single-shared weight layout")
        with torch_device_fn.device(aq.device):
            _mm_w8a8_kernel[(triton.cdiv(m, 128) * triton.cdiv(n, 128),)](
                aq,
                weight.tiled,
                scale_a,
                weight.scale,
                out,
                m,
                n,
                k,
                128,
                128,
                128,
                True,
                1,
                PACKED_B=True,
                SINGLE_SHARED=True,
                num_warps=4,
                num_stages=2,
                pipeline="basic",
                pipeline_load_num=2,
                single_shared_pipeline=True,
                single_shared_async=True,
                enable_fp_fusion=False,
            )
        return out
    with torch_device_fn.device(aq.device):
        _mm_w8a8_kernel[(triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
            aq,
            weight.tiled,
            scale_a,
            weight.scale,
            out,
            m,
            n,
            k,
            256,
            256,
            64,
            True,
            1,
            PACKED_B=True,
            num_warps=8,
            num_stages=2,
            pipeline="basic",
            enable_fp_fusion=False,
        )
    return out


def _mm_w8a8_int8_packed_weight_out(a, weight, *, out):
    """Quantize current activations on every call, including graph replay."""
    aq, scale_a = _prepare_mm_w8a8_int8_activation(a)
    return _mm_w8a8_int8_packed_prequantized_out(aq, scale_a, weight, out=out)


@libentry()
@triton.jit
def _scaled_mm_bias_kernel(
    X, BIAS, OUT, SIZE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    value = tl.load(X + offsets, mask, other=0)
    bias = tl.load(BIAS + offsets % N, mask, other=0).to(tl.float32)
    tl.store(OUT + offsets, value + bias, mask)


def _scaled_mm_arguments(a, b, scale_a, scale_b, out_dtype, bias):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be tensors")
    if a.ndim < 1 or b.ndim != 2 or a.shape[-1] != b.shape[0]:
        raise ValueError("expected A[...,K] and B[K,N]")
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("out_dtype must be FP16 or BF16, as in vLLM scaled_mm")
    k, n = b.shape
    # Explicit M avoids ambiguous reshape(-1, 0) for empty reductions.
    m = a.numel() // k if k else math.prod(a.shape[:-1])
    a2d = a.reshape(m, k)

    def normalize(scale, size, name):
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if scale.device != a.device or scale.dtype != torch.float32:
            raise ValueError(f"{name} must be FP32 on the input device")
        if scale.numel() not in (1, size):
            raise ValueError(f"{name} must be scalar or contain {size} values")
        flat = scale.reshape(-1)
        return (
            flat.expand(size).contiguous() if flat.numel() == 1 else flat.contiguous()
        )

    sa = normalize(scale_a, m, "scale_a")
    sb = normalize(scale_b, n, "scale_b")
    _validate_mm_inputs(a2d, b, sa, sb)
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a tensor")
        if bias.device != a.device or bias.dtype != out_dtype or bias.numel() != n:
            raise ValueError(
                "bias must contain N values with output dtype on the input device"
            )
        bias = bias.reshape(-1).contiguous()
    return a2d, sa, sb, bias, (*a.shape[:-1], n)


def mm_w8a8_int8(a, b, scale_a, scale_b, out_dtype, bias=None):
    """vLLM-style scaled_mm: INT8 A[...,K] @ B[K,N], scales, optional bias.

    Scales are FP32, scalar or per flattened A row / B column. Returns
    FP16/BF16 with A's leading dimensions. No input quantization or weight
    snapshot is performed. Asymmetric zero-point correction is not supported.
    """
    a2d, sa, sb, bias, shape = _scaled_mm_arguments(
        a, b, scale_a, scale_b, out_dtype, bias
    )
    out = torch.empty(shape, device=a.device, dtype=out_dtype)
    return _scaled_mm_execute(a2d, b, sa, sb, bias, out)


def _scaled_mm_execute(a, b, sa, sb, bias, out):
    m, k = a.shape
    n = b.shape[1]
    flat_out = out.view(m, n)
    if bias is None:
        _run_mm(a, b, sa, sb, flat_out, m, n, k)
    elif m and n:
        # Keep FP32 through bias addition; rounding before bias is incorrect.
        value = torch.empty((m, n), device=a.device, dtype=torch.float32)
        _run_mm(a, b, sa, sb, value, m, n, k)
        with torch_device_fn.device(a.device):
            _scaled_mm_bias_kernel[(triton.cdiv(m * n, 512),)](
                value, bias, flat_out, m * n, n, 512
            )
    return out


def mm_w8a8_int8_out(a, b, scale_a, scale_b, *, out, bias=None):
    """Caller-owned output variant of mm_w8a8_int8; dtype comes from out."""
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a tensor")
    a2d, sa, sb, bias, shape = _scaled_mm_arguments(
        a, b, scale_a, scale_b, out.dtype, bias
    )
    if out.shape != shape or out.device != a.device or not out.is_contiguous():
        raise ValueError(
            "out must be contiguous with the result shape on the input device"
        )
    return _scaled_mm_execute(a2d, b, sa, sb, bias, out)
