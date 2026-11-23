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

"""GCU400-specific W8A16 FP8 TopK.

The first bit-decoding implementation (a single kernel with K rounds of max/argmax, plus a
two-stage chunk+merge for large N) only reached ~0.45x average speedup under the benchmark
protocol: each selection round costs one reduction-tree latency (~0.2-1.2us/round, degrading with
width and residency form), and for K=64..256 the O(K*N) selection chain pushed (8,32768,256) to
738us (torch native 24.7us). This round dispatches three paths by shape:

1. ``_multi_row_kernel``: for small N, the K rounds of one row become **U-row interleaving**
   (same-phase batching: all rows max -> all rows min-pos -> all rows invalidate); the independent
   chains hide the reduction-tree latency from each other, and U is tuned by "ceil(M/U) waves =
   24 SIP" plus the "U*BLOCK_N <= 2816 lane budget"; the compare domain becomes a monotonic int32
   key of the fp32 value (two pure reductions replace max+argmax, and the -0.0/+0.0 order matches
   the torch bit-pattern order).
2. ``_dequant_bf16_kernel`` + the sibling module ``topk_mradix``: large N uses the "decode fp8
   into a bf16 row -> hardware MRADIX radix-select" combination (FlagGems own kernel, one HBM read
   plus a VDMEM digital scan, measured 22-38us across the N=4k..32k tiers).
   largest=False flips the fp8 sign bit while decoding (equivalent to negating the value) and
   restores it with a small post negate kernel; the reference domain is also bf16 dequant, so the
   value ranges match exactly.
3. The existing single-kernel / two-stage chunk+merge paths remain as the fallback outside the
   MRADIX envelope (N>65536, K>512, non-bf16 out_dtype, etc.).
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

try:
    # Compositional reuse of the sibling module's hardware MRADIX topk (TLE-Struct + TLE-Raw). If the import fails
    # (no .tops file / no TLE-Raw binding), silently fall back to the old path.
    from . import topk_mradix as _mradix

    _MRADIX_OK = bool(_mradix.HAS_TLE_RAW)
except Exception:  # noqa: BLE001
    _mradix = None
    _MRADIX_OK = False

# Single-block column width: fp32 candidate live set 4096 * 4B = 16KB, consistent with the rms_norm tile budget.
_CHUNK = 4096
# Candidate buffer cap (NUM_CHUNK * K per row); beyond it, increase the block width to reduce the number of blocks.
_CAND_CAP = 8192
# Minimum reduction amount (number of blocks * K) to enable two-stage; below it, use a single kernel (saves one launch).
_CHUNK_MIN_REDUCTIONS = 1024
# Beyond this block width, abandon two-stage and fall back to a single kernel (avoid ultra-wide vectors entering the DSM
# wrong-value region).
_MAX_CHUNK = 65536
_CHUNK_WARPS = 4
_MERGE_WARPS = 1
_SINGLE_WARPS = 1

# Settings for row-interleaving over K rounds (2026-11-18 U-sweep measurements: with U=1 and no interleaving, the
# dependency chain per round
# takes ~500-700ns; with U>=2 interleaving, ~110-170ns (7-20x); optimal U = the smallest power of 2 that fills one full
# wave (24 SIPs),
# with U*BLOCK_N capped at 8192 lane ((128,1024,16) with U=8 measured best).
_KR_WAVES = 24
_KR_LANE_BUDGET = 8192
# Cost model per round per row (us): <=1024 width resident ~0.28, wider scales x2 by byte ratio (spill penalty).
_KR_ROUND_US = 0.28
# MRADIX per-row cost model (us): fixed ~9 + N * 0.7ns (VDMEM digit scan).
_MR_ROW_FIXED_US = 9.0
_MR_ROW_PER_ELEM_NS = 0.0007
# mradix envelope (consistent with topk_mradix.TOPK_K_MAX / SORT_K_MAX / L1_MAX_COLS).
_MR_K_MAX = 512
_MR_N_MAX = 65536


@triton.jit
def _decode_fp8(bits, E4M3: tl.constexpr):
    """FP8 bit pattern -> float32 (subnormals included; no device-side FP8 type needed)."""
    sign = (bits >> 7) & 1
    if E4M3:
        exp = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        normal_bits = (sign << 31) | ((exp + 120) << 23) | (mantissa << 20)
        subnormal = mantissa.to(tl.float32) * 0.001953125
    else:
        exp = (bits >> 2) & 0x1F
        mantissa = bits & 0x3
        normal_bits = (sign << 31) | ((exp + 112) << 23) | (mantissa << 21)
        subnormal = mantissa.to(tl.float32) * 1.52587890625e-5

    # The IEEE bits of a normal already include the sign bit; a subnormal has no bit pattern, so the sign must be
    # carried separately
    normal_value = normal_bits.to(tl.float32, bitcast=True)
    signed_subnormal = tl.where(sign == 1, -subnormal, subnormal)
    return tl.where(exp > 0, normal_value, signed_subnormal)


@triton.jit
def _group_index(cols, GROUP_SIZE: tl.constexpr, LOG_GROUP: tl.constexpr):
    """Group index: shift for a power of two, else division (GCU has no integer divide, per-lane expansion is costly)."""
    if LOG_GROUP > 0:
        return cols >> LOG_GROUP
    else:
        return cols // GROUP_SIZE


@triton.jit
def _mono_key(value):
    """fp32 -> monotonic int32 key (invertible by itself).

    when bits>=0 the key is bits, otherwise key=bits^0x7FFFFFFF: the negative range maps back into
    [-2^31, -1], so the order of any two values matches the key order, and -0.0 < +0.0 (matching
    the bit-pattern order of torch.topk). Flipping the sign (~key) gives the descending domain.
    """
    bits = value.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _unmono_key(key):
    bits = tl.where(key >= 0, key, key ^ 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True)


_KEY_SENTINEL = tl.constexpr(-2147483648)


@libentry()
@triton.jit
def topk_w8a16_fp8_multi_row_kernel(
    x_ptr,
    scale_ptr,
    y_vals_ptr,
    y_idx_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LARGEST: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG_GROUP: tl.constexpr,
    U: tl.constexpr,
):
    """U-row interleaved K-round selection: same-phase batching hides the reduction-tree latency.

    Each round does, on the int32 key domain of [U, BLOCK_N]: all rows max -> all rows min-position
    (smallest column among ties) -> vector store -> all rows invalidate.
    """
    pid = tl.program_id(0)
    rows = pid * U + tl.arange(0, U)
    live = rows < M
    cols = tl.arange(0, BLOCK_N)
    valid = (cols < N)[None, :] & live[:, None]
    bits = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=valid, other=0).to(
        tl.int32
    )
    decoded = _decode_fp8(bits, E4M3)
    group = _group_index(cols, GROUP_SIZE, LOG_GROUP)[None, :]
    scale = tl.load(scale_ptr + rows[:, None] * GROUPS + group, mask=valid, other=0).to(
        tl.float32
    )
    key = _mono_key(decoded * scale)
    if not LARGEST:
        key = ~key
    cand = tl.where(valid, key, _KEY_SENTINEL)

    for i in range(K):
        best = tl.max(cand, axis=1)
        pos = tl.min(tl.where(cand == best[:, None], cols[None, :], BLOCK_N), axis=1)
        if not LARGEST:
            # After flipping the key domain, max selects the smallest-value element; flipping back to the original
            # domain restores the true value
            best = ~best
        vals = _unmono_key(best)
        tl.store(
            y_vals_ptr + rows * K + i, vals.to(y_vals_ptr.dtype.element_ty), mask=live
        )
        tl.store(y_idx_ptr + rows * K + i, pos.to(tl.int64), mask=live)
        cand = tl.where(cols[None, :] == pos[:, None], _KEY_SENTINEL, cand)


@libentry()
@triton.jit
def topk_w8a16_fp8_kernel(
    x_ptr,
    scale_ptr,
    y_vals_ptr,
    y_idx_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LARGEST: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG_GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    valid = offs < N
    bits = tl.load(x_ptr + row * N + offs, mask=valid, other=0).to(tl.int32)
    decoded = _decode_fp8(bits, E4M3)
    group = _group_index(offs, GROUP_SIZE, LOG_GROUP)
    scale = tl.load(scale_ptr + row * GROUPS + group, mask=valid, other=0).to(
        tl.float32
    )
    value = decoded * scale

    # Mask invalid padding with extrema so extrema / indices are selected only among real elements
    if LARGEST:
        sentinel = float("-inf")
    else:
        sentinel = float("inf")
    candidates = tl.where(valid, value, sentinel)

    for i in range(K):
        if LARGEST:
            best_value = tl.max(candidates, 0)
            best_index = tl.argmax(candidates, 0)
        else:
            best_value = tl.min(candidates, 0)
            best_index = tl.argmin(candidates, 0)
        tl.store(y_vals_ptr + row * K + i, best_value)
        tl.store(y_idx_ptr + row * K + i, best_index.to(tl.int64))
        candidates = tl.where(offs == best_index, sentinel, candidates)


@triton.jit
def topk_w8a16_fp8_chunk_kernel(
    x_ptr,
    scale_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    N,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LARGEST: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG_GROUP: tl.constexpr,
    NUM_CHUNK: tl.constexpr,
):
    """Stage 1: one program owns a (row, block) pair and emits that block's top-K candidates."""
    pid = tl.program_id(0)
    row = pid // NUM_CHUNK
    chunk = pid - row * NUM_CHUNK
    offs = tl.arange(0, BLOCK_N)
    cols = chunk * BLOCK_N + offs
    valid = cols < N

    bits = tl.load(x_ptr + row * N + cols, mask=valid, other=0).to(tl.int32)
    decoded = _decode_fp8(bits, E4M3)
    group = _group_index(cols, GROUP_SIZE, LOG_GROUP)
    scale = tl.load(scale_ptr + row * GROUPS + group, mask=valid, other=0).to(
        tl.float32
    )
    if LARGEST:
        sentinel = float("-inf")
    else:
        sentinel = float("inf")
    candidates = tl.where(valid, decoded * scale, sentinel)

    # Candidates are packed tightly in program order: pid * K + i == row*NUM_CHUNK*K + chunk*K + i
    for i in range(K):
        if LARGEST:
            best_value = tl.max(candidates, 0)
        else:
            best_value = tl.min(candidates, 0)
        # For equal values, take the smallest lane, ensuring that together with the later "hole punching" the same
        # element is never selected twice
        pos = tl.min(tl.where(candidates == best_value, offs, BLOCK_N), 0)
        tl.store(cand_val_ptr + pid * K + i, best_value)
        tl.store(cand_idx_ptr + pid * K + i, (chunk * BLOCK_N + pos).to(tl.int32))
        candidates = tl.where(offs == pos, sentinel, candidates)


@triton.jit
def topk_w8a16_fp8_merge_kernel(
    cand_val_ptr,
    cand_idx_ptr,
    y_vals_ptr,
    y_idx_ptr,
    K: tl.constexpr,
    LARGEST: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_CAND: tl.constexpr,
):
    """Stage 2: takes K extremes again over each row's NUM_CAND candidates, emitting the sorted top-K."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    valid = offs < NUM_CAND
    if LARGEST:
        sentinel = float("-inf")
    else:
        sentinel = float("inf")
    values = tl.load(cand_val_ptr + row * NUM_CAND + offs, mask=valid, other=sentinel)
    indices = tl.load(cand_idx_ptr + row * NUM_CAND + offs, mask=valid, other=-1)
    candidates = tl.where(valid, values, sentinel)

    for i in range(K):
        if LARGEST:
            best_value = tl.max(candidates, 0)
        else:
            best_value = tl.min(candidates, 0)
        pos = tl.min(tl.where(candidates == best_value, offs, BLOCK_C), 0)
        # Retrieve the true column index of this candidate (scalar reduction, only one value is moved)
        best_index = tl.sum(tl.where(offs == pos, indices, 0), 0)
        tl.store(y_vals_ptr + row * K + i, best_value)
        tl.store(y_idx_ptr + row * K + i, best_index.to(tl.int64))
        candidates = tl.where(offs == pos, sentinel, candidates)


@libentry()
@triton.jit
def topk_w8a16_fp8_dequant_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    N,
    GROUPS,
    GROUP_SIZE: tl.constexpr,
    E4M3: tl.constexpr,
    NEGATE: tl.constexpr,
    GT: tl.constexpr,
):
    """fp8 + group scale -> bf16 row (stage 1 of the MRADIX combination path).

    Memory is accessed as a [GT, GROUP_SIZE] 2-D group tile: x/out use affine contiguous blocks and
    the scale is a single [GT] 1-D load broadcast afterwards -- avoiding per-element gather-style
    scale loads (element-wise issue costs ~10ns/element on GCU, measured 2026-11-18, which dragged
    this kernel to ~21us). With NEGATE the fp8 sign bit is flipped (equivalent to negating the
    value), letting largest=False reuse the largest-only radix-select of MRADIX. Requires
    group_size to be a power of two (gated on the host side).
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    g = tile * GT + tl.arange(0, GT)
    gvalid = g < GROUPS
    cols = tl.arange(0, GROUP_SIZE)
    inc = g[:, None] * GROUP_SIZE + cols[None, :]
    valid = inc < N
    bits = tl.load(x_ptr + row * N + inc, mask=valid, other=0).to(tl.int32)
    if NEGATE:
        bits = bits ^ 0x80
    decoded = _decode_fp8(bits, E4M3)
    scale = tl.load(scale_ptr + row * GROUPS + g, mask=gvalid, other=0).to(tl.float32)
    val = decoded * scale[:, None]
    tl.store(out_ptr + row * N + inc, val.to(out_ptr.dtype.element_ty), mask=valid)


@libentry()
@triton.jit
def topk_w8a16_fp8_negate_kernel(
    y_ptr,
    total,
    BLOCK: tl.constexpr,
):
    """Negates the (M, K) value tensor element-wise (the restore after the largest=False detour)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < total
    vals = tl.load(y_ptr + offs, mask=valid)
    tl.store(y_ptr + offs, -vals, mask=valid)


def _log2_if_pow2(value):
    """Returns log2 for a power of two, else 0 (the kernel picks shift or division from it)."""
    if value <= 0 or (value & (value - 1)) != 0:
        return 0
    return value.bit_length() - 1


def _kround_rows(m, block_n, k):
    """U tuning for the row-interleaved K rounds.

    U is the smallest power of two that fills one wave (ceil(M/U) <= 24 SIP), floored at 2 (with
    U=1 and no interleaving the dependency chain degrades 4-7x per round, measured 2026-11-18);
    when U*BLOCK_N exceeds 8192 lanes it is halved to prevent spills. Returns (U, estimated makespan).
    """
    u = 2
    while u * _KR_WAVES < m:
        u *= 2
    while u > 2 and u * block_n > _KR_LANE_BUDGET:
        u //= 2
    programs = -(-m // u)
    waves = max(1, -(-programs // _KR_WAVES))
    per_round = _KR_ROUND_US if block_n <= 1024 else _KR_ROUND_US * block_n / 1024 * 2
    return u, waves * u * k * per_round


def _mradix_rows_us(m, n):
    waves = max(1, -(-m // _KR_WAVES))
    return waves * (_MR_ROW_FIXED_US + n * _MR_ROW_PER_ELEM_NS)


def _chunk_plan(n, k):
    """Returns (block width, block count); (None, 1) to use the single kernel when not worth it or too wide.

    Measured (2026-11-17, gcu400): the two-stage total time is roughly proportional to "block count
    x K" reductions plus one extra launch, so it is enabled only when the reduction volume is large
    enough (``block count x K >= 1024``):
    * (8,32768,256): 8 blocks x 256 = 2048 -> two-stage 1.22ms (single kernel 1.62ms);
    * (16,16384,128): 4 x 128 = 512 -> single kernel 0.58ms is faster (two-stage 0.59ms);
    * (32,8192,64): 2 x 64 = 128 -> single kernel 0.36ms.

    The block width is only doubled when the candidate buffer exceeds ``_CAND_CAP``.
    """
    chunk = _CHUNK
    while True:
        num_chunk = triton.cdiv(n, chunk)
        if num_chunk * k <= _CAND_CAP or chunk >= _MAX_CHUNK:
            break
        chunk *= 2
    if num_chunk <= 1 or chunk > _MAX_CHUNK or num_chunk * k < _CHUNK_MIN_REDUCTIONS:
        return None, 1
    return chunk, num_chunk


def _topk_kround(
    x_bits,
    scale_2d,
    y_vals,
    y_idx,
    m,
    n,
    k,
    num_groups,
    group_size,
    largest,
    is_e4m3,
    log_group,
):
    block_n = triton.next_power_of_2(n)
    u, _ = _kround_rows(m, block_n, k)
    with torch_device_fn.device(x_bits.device):
        topk_w8a16_fp8_multi_row_kernel[(-(-m // u),)](
            x_bits,
            scale_2d,
            y_vals.reshape(m, k),
            y_idx.reshape(m, k),
            m,
            n,
            k,
            num_groups,
            group_size,
            largest,
            is_e4m3,
            block_n,
            log_group,
            u,
            num_warps=1,
        )
    return y_vals, y_idx


def _topk_mradix_compose(
    x_bits,
    scale_2d,
    y_vals,
    y_idx,
    m,
    n,
    k,
    num_groups,
    group_size,
    largest,
    is_e4m3,
    log_group,
):
    """Decode fp8 into a bf16 row -> hardware MRADIX topk (largest=False detours through negation).

    It calls the sibling module's launcher directly instead of its Python wrapper: skipping the
    internal i32 index tensor and the `.to(int64)` aten cast saves one kernel plus dispatch
    overhead, and the i64 output lands via the launcher's implicit widening store.
    """
    with torch_device_fn.device(x_bits.device):
        dequant = torch.empty((m, n), device=x_bits.device, dtype=torch.bfloat16)
        # GT adaptation: larger tiles amortize the per-program fixed cost (2026-11-18 measurements:
        # GT 8/32/128 = 30/12.5/10.5us@N=32768), capped at 128 groups to avoid register pressure.
        gt = min(128, triton.next_power_of_2(num_groups))
        topk_w8a16_fp8_dequant_kernel[(m, triton.cdiv(num_groups, gt))](
            x_bits,
            scale_2d,
            dequant,
            n,
            num_groups,
            group_size,
            is_e4m3,
            not largest,
            gt,
            num_warps=4,
        )
        grid = min(_mradix._max_grid(x_bits.device.index), m)
        _mradix.topk_mradix_l1_launch[(grid,)](
            dequant,
            y_vals.reshape(m, k),
            y_idx.reshape(m, k),
            m,
            n,
            K=k,
            K_PAD=triton.next_power_of_2(k),
            BLOCK_N=max(256, triton.next_power_of_2(n)),
            ROW_STEP=1,
            SORTED=1,
            IN_KIND=2,
            OUT_KIND=2,
            num_warps=4,
            num_stages=1,
        )
        if not largest:
            topk_w8a16_fp8_negate_kernel[(triton.cdiv(m * k, 1024),)](
                y_vals, m * k, 1024, num_warps=4
            )
    return y_vals, y_idx


def topk_w8a16_fp8(
    x_fp8,
    x_scale,
    k,
    dim=-1,
    largest=True,
    sorted=True,
    group_size=128,
    out_dtype=torch.bfloat16,
):
    logger.debug("GEMS_ENFLAME TOPK_W8A16_FP8")
    if dim < 0:
        dim += x_fp8.ndim
    assert dim == x_fp8.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"

    leading_shape = x_fp8.shape[:-1]
    n = x_fp8.shape[-1]
    if k == 0:
        out_shape = list(leading_shape) + [0]
        return (
            torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype),
            torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64),
        )

    batch_size = math.prod(leading_shape) if leading_shape else 1
    num_groups = triton.cdiv(n, group_size)
    expected_scale_shape = tuple(leading_shape) + (num_groups,)
    assert (
        x_scale.shape == expected_scale_shape
    ), f"x_scale shape should be {expected_scale_shape}, got {x_scale.shape}"

    # Read FP8 bit patterns as uint8: does not rely on device-side FP8 pointers, nor does it modify the quantized data
    x_bits = x_fp8.contiguous().view(torch.uint8).reshape(batch_size, n)
    scale_2d = x_scale.reshape(batch_size, num_groups)
    is_e4m3 = x_fp8.dtype == torch.float8_e4m3fn
    log_group = _log2_if_pow2(group_size)

    def alloc_out():
        return (
            torch.empty(
                tuple(leading_shape) + (k,), device=x_fp8.device, dtype=out_dtype
            ),
            torch.empty(
                tuple(leading_shape) + (k,), device=x_fp8.device, dtype=torch.int64
            ),
        )

    # Path 1: row-interleaved K rounds (small N / small K: the O(K) selection chain is fastest at resident width).
    if n <= _KR_LANE_BUDGET:
        block_n = triton.next_power_of_2(n)
        _, est_kround = _kround_rows(batch_size, block_n, k)
        use_kround = est_kround <= _mradix_rows_us(batch_size, n) + 4.0
    else:
        use_kround = False
    if use_kround:
        y_vals, y_idx = alloc_out()
        return _topk_kround(
            x_bits,
            scale_2d,
            y_vals,
            y_idx,
            batch_size,
            n,
            k,
            num_groups,
            group_size,
            largest,
            is_e4m3,
            log_group,
        )

    # Path 2: dequant + hardware MRADIX (workhorse for large N; K/N within the envelope, bf16 output,
    # group_size a power of two — required by dequant's 2D group-tile memory access).
    if (
        _MRADIX_OK
        and 0 < k <= _MR_K_MAX
        and n <= _MR_N_MAX
        and out_dtype == torch.bfloat16
        and log_group > 0
    ):
        y_vals, y_idx = alloc_out()
        return _topk_mradix_compose(
            x_bits,
            scale_2d,
            y_vals,
            y_idx,
            batch_size,
            n,
            k,
            num_groups,
            group_size,
            largest,
            is_e4m3,
            log_group,
        )

    # Path 3 (fallback): single kernel / two-stage chunk+merge.
    y_vals, y_idx = alloc_out()
    with torch_device_fn.device(x_fp8.device):
        chunk, num_chunk = _chunk_plan(n, k)
        if chunk is None:
            topk_w8a16_fp8_kernel[(batch_size,)](
                x_bits,
                scale_2d,
                y_vals.reshape(batch_size, k),
                y_idx.reshape(batch_size, k),
                n,
                k,
                num_groups,
                group_size,
                largest,
                is_e4m3,
                triton.next_power_of_2(n),
                log_group,
                num_warps=_SINGLE_WARPS,
            )
        else:
            num_cand = num_chunk * k
            cand_val = torch.empty(
                (batch_size * num_cand,), device=x_fp8.device, dtype=torch.float32
            )
            cand_idx = torch.empty(
                (batch_size * num_cand,), device=x_fp8.device, dtype=torch.int32
            )
            topk_w8a16_fp8_chunk_kernel[(batch_size * num_chunk,)](
                x_bits,
                scale_2d,
                cand_val,
                cand_idx,
                n,
                k,
                num_groups,
                group_size,
                largest,
                is_e4m3,
                chunk,
                log_group,
                num_chunk,
                num_warps=_CHUNK_WARPS,
            )
            topk_w8a16_fp8_merge_kernel[(batch_size,)](
                cand_val,
                cand_idx,
                y_vals.reshape(batch_size, k),
                y_idx.reshape(batch_size, k),
                k,
                largest,
                triton.next_power_of_2(num_cand),
                num_cand,
                num_warps=_MERGE_WARPS,
            )
    return y_vals, y_idx
