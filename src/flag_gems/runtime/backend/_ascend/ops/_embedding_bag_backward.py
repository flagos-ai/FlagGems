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

import torch
import triton
import triton.language as tl
from packaging import version

from flag_gems.ops._embedding_bag_backward import (
    _eb_backward_init,
    _eb_backward_scatter,
    _eb_backward_scatter_body,
    _eb_backward_sparse,
    _eb_backward_validate,
    _embedding_bag_backward_impl,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_LEGACY_TRITON = version.parse(triton.__version__) < version.parse("3.5")


@triton.jit
def _round_positive_reduced(value, BF16: tl.constexpr):
    bits = value.to(tl.int32, bitcast=True)
    if BF16:
        bias = 0x7FFF + ((bits >> 16) & 1)
        result = ((bits + bias) & -65536).to(tl.float32, bitcast=True)
    else:
        # FP16 normals retain ten fraction bits. Handle overflow to infinity
        # and subnormals separately, including a zero reciprocal after overflow.
        rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & -8192
        normal = tl.where(rounded >= 0x47800000, 0x7F800000, rounded)
        subnormal = bits < 0x38800000
        scaled = tl.where(subnormal, value, 0.0) * 16777216.0
        integral = scaled.to(tl.int32)
        fraction = scaled - integral.to(tl.float32)
        increment = (fraction > 0.5) | ((fraction == 0.5) & ((integral & 1) != 0))
        rounded_subnormal = (integral + increment.to(tl.int32)).to(tl.float32) * (
            1.0 / 16777216.0
        )
        result = tl.where(
            subnormal, rounded_subnormal, normal.to(tl.float32, bitcast=True)
        )
    return result


@libentry()
@triton.jit
def _embedding_bag_sparse_mean(
    GRAD,
    INDICES,
    MAPPING,
    BAG_SIZE,
    WEIGHTS,
    PREFIX,
    CHUNKS,
    OUT_IDX,
    VALUES,
    N,
    B,
    D,
    V,
    PAD,
    SG0: tl.constexpr,
    SG1: tl.constexpr,
    SI: tl.constexpr,
    SB: tl.constexpr,
    SW: tl.constexpr,
    MODE: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    COMPACT: tl.constexpr,
    RANKED: tl.constexpr,
    FP64: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    nd = tl.maximum(tl.cdiv(D, BD), 1)
    sample = (pid // nd) * BS + tl.arange(0, BS)
    col = (pid % nd) * BD + tl.arange(0, BD)
    idx = tl.load(INDICES + sample * SI, sample < N, other=PAD)
    active = (sample < N) & (idx >= 0) & (idx < V) & (idx != PAD)
    if COMPACT:
        pos = tl.load(PREFIX + sample, sample < N, other=0) - 1
        if RANKED:
            chunk = sample // 256
            pos += tl.load(CHUNKS + chunk, sample < N, other=0)
    else:
        pos = sample
    pos = pos.to(tl.int64)
    bag = tl.load(MAPPING + sample, sample < N, other=0)
    active = active & (bag >= 0) & (bag < B)
    if FP64:
        acc_type = tl.float64
    else:
        acc_type = tl.float32
    g = tl.load(
        GRAD + bag[:, None] * SG0 + col[None, :] * SG1,
        active[:, None] & (col[None, :] < D),
        other=0,
    ).to(acc_type)
    if MODE == 1:
        # ATen materializes the inverse bag size in grad dtype before its
        # sparse per-occurrence multiplication, including the reduced-dtype
        # rounding of both the denominator and reciprocal.
        size = tl.load(BAG_SIZE + bag * SB, active, other=1)
        # Materialize the reduced-dtype RNE stages with integer operations.
        # A cast to the input dtype and back can otherwise be folded away.
        reduced_bf16: tl.constexpr = GRAD.dtype.element_ty == tl.bfloat16
        size = _round_positive_reduced(tl.maximum(size, 1).to(tl.float32), reduced_bf16)
        inverse = _round_positive_reduced(1.0 / tl.maximum(size, 1), reduced_bf16)
        g = g * inverse[:, None]
    if HAS_WEIGHTS:
        weight = tl.load(WEIGHTS + sample * SW, sample < N, other=0).to(acc_type)
        g = g * weight[:, None]
    if pid % nd == 0:
        tl.store(OUT_IDX + pos, idx, active)
    tl.store(
        VALUES + pos[:, None] * D + col[None, :],
        g,
        active[:, None] & (col[None, :] < D),
    )


@libentry()
@triton.jit
def _legacy_offset2bag(
    OFFSETS,
    MAPPING,
    N: tl.constexpr,
    B: tl.constexpr,
    SO: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    low = tl.full((BLOCK,), 0, tl.int64)
    high = tl.full((BLOCK,), B, tl.int64)
    for _ in range(STEPS):
        mid = (low + high) // 2
        boundary = tl.load(OFFSETS + mid * SO, (x < N) & (mid < B), other=N)
        right = (mid < B) & (boundary <= x)
        low = tl.where(right, mid + 1, low)
        high = tl.where(right, high, mid)
    tl.store(MAPPING + x, tl.maximum(low - 1, 0), x < N)


@libentry()
@triton.jit
def _legacy_scatter(
    GRAD,
    INDICES,
    MAPPING,
    BAG_SIZE,
    MAXIMUM,
    WEIGHTS,
    ACC,
    N: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    PAD: tl.constexpr,
    SG0: tl.constexpr,
    SG1: tl.constexpr,
    SI: tl.constexpr,
    SB: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SW: tl.constexpr,
    MODE: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    FP64: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
):
    # CANN 8.5's pointer analysis cannot reliably lower these dynamic strides.
    # Fix their metadata at compilation while retaining the generic arithmetic.
    _eb_backward_scatter_body(
        GRAD,
        INDICES,
        MAPPING,
        BAG_SIZE,
        MAXIMUM,
        WEIGHTS,
        ACC,
        N,
        B,
        D,
        V,
        PAD,
        SG0,
        SG1,
        SI,
        SB,
        SX0,
        SX1,
        SW,
        MODE,
        HAS_WEIGHTS,
        FP64,
        BD,
        BS,
    )


@libentry()
@triton.jit
def _legacy_sparse_zero_dim_indices(
    INDICES,
    MAPPING,
    PREFIX,
    CHUNKS,
    OUT_IDX,
    N: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    PAD: tl.constexpr,
    SI: tl.constexpr,
    COMPACT: tl.constexpr,
    RANKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Empty feature dimensions need only the COO row indices. In old CANN,
    # even fully masked floating-point accesses can touch an empty allocation.
    lane = tl.arange(0, BLOCK)
    for base in range(
        tl.program_id(0).to(tl.int64) * BLOCK,
        N,
        tl.num_programs(0).to(tl.int64) * BLOCK,
    ):
        sample = base.to(tl.int64) + lane
        idx = tl.load(INDICES + sample * SI, sample < N, other=PAD)
        bag = tl.load(MAPPING + sample, sample < N, other=0)
        active = (sample < N) & (idx >= 0) & (idx < V) & (idx != PAD)
        active = active & (bag >= 0) & (bag < B)
        if COMPACT:
            pos = tl.load(PREFIX + sample, sample < N, other=0).to(tl.int64) - 1
            if RANKED:
                chunk = tl.load(CHUNKS + sample // 256, sample < N, other=0)
                pos += chunk.to(tl.int64)
        else:
            pos = sample
        tl.store(OUT_IDX + pos, idx, active)


@libentry()
@triton.jit
def _legacy_empty_accumulator_init(
    FREQ, ERROR, V: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr
):
    # The accumulator has no elements; only real metadata buffers need writes.
    if COUNT and V > 0:
        x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        tl.store(FREQ + x, 0, x < V)
    if tl.program_id(0) == 0:
        tl.store(ERROR, 0)


@libentry()
@triton.jit
def _legacy_empty_indices_validate(
    OFFSETS,
    BAG_SIZE,
    MAXIMUM,
    ERROR,
    B: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    O: tl.constexpr,
    SO: tl.constexpr,
    SB: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    MODE: tl.constexpr,
    SPARSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # With N=0 no index, mapping, frequency, or compaction storage is accessed.
    # Keep all remaining offset/size/MAX checks, guarded by nonempty extents.
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    invalid = tl.full((BLOCK,), 0, tl.int32)
    if O > 0:
        off = tl.load(OFFSETS + x * SO, x < O, other=0)
        invalid |= tl.where((x < O) & (off != 0), 2, 0)
    if B > 0 and D > 0:
        if MODE == 1 or (MODE == 2 and not SPARSE):
            size = tl.load(BAG_SIZE + x * SB, x < B, other=0)
            invalid |= tl.where((x < B) & (size != 0), 8, 0)
            if MODE == 2:
                row, col = x // D, x % D
                maximum = tl.load(MAXIMUM + row * SX0 + col * SX1, x < B * D, other=-1)
                bad = (x < B * D) & ((maximum < -1) | (maximum >= tl.maximum(V, 1)))
                invalid |= tl.where(bad, 16, 0)
    tl.atomic_max(ERROR, tl.max(invalid, 0), sem="relaxed")


def _launch_ascend_backward(kernel, packed_kernel, grid, pointers, metadata, **options):
    if _LEGACY_TRITON and kernel is _eb_backward_scatter and metadata[3] == 0:
        # There are no output rows. The validator still reports every invalid
        # index, but old CANN must not launch atomics with an empty accumulator.
        return None
    if _LEGACY_TRITON and kernel is _eb_backward_validate and metadata[4] == 0:
        # No row can contribute a frequency when the table has zero rows.
        # Keep index checks and avoid masked atomics to an empty count buffer.
        metadata = (*metadata[:16], False, *metadata[17:])
    if _LEGACY_TRITON and kernel is _eb_backward_init and metadata[0] == 0:
        return _legacy_empty_accumulator_init[grid](
            pointers[1], pointers[2], metadata[1], metadata[2], metadata[3]
        )
    if _LEGACY_TRITON and kernel is _eb_backward_validate and metadata[1] == 0:
        return _legacy_empty_indices_validate[grid](
            pointers[1],
            pointers[3],
            pointers[4],
            pointers[9],
            metadata[2],
            metadata[3],
            metadata[4],
            metadata[5],
            metadata[8],
            metadata[10],
            metadata[11],
            metadata[12],
            metadata[14],
            metadata[15],
            metadata[21],
        )
    if _LEGACY_TRITON and kernel is _eb_backward_sparse and metadata[2] == 0:
        # The generic validator and exact COO size calculation have already run.
        # This launch never receives or touches a zero-sized floating buffer.
        return _legacy_sparse_zero_dim_indices[
            (min(triton.cdiv(metadata[0], 128), 65535),)
        ](
            pointers[1],
            pointers[2],
            pointers[5],
            pointers[6],
            pointers[7],
            metadata[0],
            metadata[1],
            metadata[3],
            metadata[4],
            metadata[7],
            metadata[12],
            metadata[13],
            128,
        )
    if _LEGACY_TRITON and kernel is _eb_backward_validate and not metadata[13]:
        # Reuse the existing disjoint mapping allocation after host validation.
        # The fixed-step search is the one already used by the forward kernel;
        # invalid offsets are still reported by the unchanged validator.
        n, b = metadata[1:3]
        mapping = pointers[5]
        if n:
            _legacy_offset2bag[(triton.cdiv(n, 128),)](
                pointers[1], mapping, n, b, metadata[8], b.bit_length(), 128
            )
        pointers = (*pointers[:2], mapping, *pointers[3:])
        metadata = (*metadata[:9], 1, *metadata[10:13], True, *metadata[14:])
    if (
        kernel is _eb_backward_scatter
        and metadata[12] == 2
        and pointers[4].dtype == torch.int64
    ):
        # CANN's int64-indexed, masked MAX scatter loses contributions with a
        # four-bag, 64-feature tile. A smaller feature tile preserves the full
        # index width and the existing FP32 accumulation contract.
        block_d = min(metadata[15], 32)
        block_s = metadata[16]
        grid = (triton.cdiv(metadata[1], block_s) * triton.cdiv(metadata[2], block_d),)
        metadata = (*metadata[:15], block_d, block_s)
    elif (
        kernel is _eb_backward_sparse
        and metadata[10] == 1
        and pointers[0].dtype in (torch.float16, torch.bfloat16)
    ):
        kernel = _embedding_bag_sparse_mean
    if _LEGACY_TRITON and kernel is _eb_backward_scatter:
        kernel = _legacy_scatter
    return kernel[grid](*pointers, *metadata, **options)


def _embedding_bag_backward(
    grad,
    indices,
    offsets,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights=None,
    padding_idx=-1,
):
    logger.debug("GEMS _EMBEDDING_BAG_BACKWARD")
    return _embedding_bag_backward_impl(
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        padding_idx,
        device_assert_enabled=False,
        fused_init_enabled=False,
        launch_fn=_launch_ascend_backward,
    )
