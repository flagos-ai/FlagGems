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

import builtins
import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
pow = tl_extra_shim.pow


def heur_block_m(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))


def heur_block_n(args):
    return builtins.min(args["N"], 8192)


@libentry()
@triton.jit
def zero_workspace_kernel(X, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(X + offsets, 0.0)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def l2_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a * a
    sum = tl.sum(_sum, axis=1)

    out = tl.sqrt(sum)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def l2_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(x * x)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l2_norm_tail_kernel(
    X,
    Mid,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    MID_INDEX: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    total = 0.0
    full_size = (TAIL_SIZE // 8) * 8
    for offset in tl.range(0, full_size, 8):
        values = tl.load(X + TAIL_OFFSET + offset + tl.arange(0, 8)).to(tl.float32)
        total += tl.sum(values * values)
    for offset in tl.static_range(TAIL_SIZE % 8):
        value = tl.load(X + TAIL_OFFSET + full_size + offset).to(tl.float32)
        total += value * value
    tl.store(Mid + MID_INDEX, total)


@libentry()
@triton.jit
def l2_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def max_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _max = tl.maximum(tl.abs(a), _max)

    max = tl.max(_max, axis=1)
    out = max[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def max_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.max(tl.abs(x))
    tl.store(Mid, mid)


@libentry()
@triton.jit
def max_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.max(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def min_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=float("inf")).to(tl.float32)
        _min = tl.minimum(tl.abs(a), _min)

    min = tl.min(_min, axis=1)
    out = min[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def min_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=float("inf")).to(tl.float32)
    mid = tl.min(tl.abs(x))
    tl.store(Mid, mid)


@libentry()
@triton.jit
def min_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=float("inf")).to(tl.float32)
    out = tl.min(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def l0_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0).to(tl.float32)
        _sum += tl.where(a != 0, 1, 0)
    sum = tl.sum(_sum, axis=1)
    out = sum[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def l0_norm_kernel_1(
    X, Mid, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    cnt = (x != 0).to(tl.float32)
    mid = tl.sum(cnt)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l0_norm_tail_kernel(
    X,
    Mid,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    MID_INDEX: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    total = 0.0
    full_size = (TAIL_SIZE // 8) * 8
    for offset in tl.range(0, full_size, 8):
        values = tl.load(X + TAIL_OFFSET + offset + tl.arange(0, 8)).to(tl.float32)
        total += tl.sum((values != 0).to(tl.float32))
    for offset in tl.static_range(TAIL_SIZE % 8):
        value = tl.load(X + TAIL_OFFSET + full_size + offset).to(tl.float32)
        total += (value != 0).to(tl.float32)
    tl.store(Mid + MID_INDEX, total)


@libentry()
@triton.jit
def l0_norm_kernel_2(
    Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sum(mid)
    tl.store(Out, out)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_M": heur_block_m,
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ord = ord.to(tl.float32)
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += pow(tl.abs(a), ord)
    sum = tl.sum(_sum, axis=1)
    out = pow(sum, 1 / ord)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_1(
    X, Mid, ord, M, BLOCK_SIZE: tl.constexpr, buffer_size_limit: tl.constexpr
):
    ord = ord.to(tl.float32)
    pid = ext.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(pow(tl.abs(x), ord))
    tl.store(Mid, mid)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_tail_kernel(
    X,
    Mid,
    ord,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    MID_INDEX: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    ord = ord.to(tl.float32)
    total = 0.0
    full_size = (TAIL_SIZE // 8) * 8
    for offset in tl.range(0, full_size, 8):
        values = tl.load(X + TAIL_OFFSET + offset + tl.arange(0, 8)).to(tl.float32)
        total += tl.sum(pow(tl.abs(values), ord))
    for offset in tl.static_range(TAIL_SIZE % 8):
        value = tl.load(X + TAIL_OFFSET + full_size + offset).to(tl.float32)
        total += pow(tl.abs(value), ord)
    tl.store(Mid + MID_INDEX, total)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_2(
    Mid, Out, ord, MID_SIZE, BLOCK_MID: tl.constexpr, buffer_size_limit: tl.constexpr
):
    ord = ord.to(tl.float32)
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = pow(tl.sum(mid), 1 / ord)
    tl.store(Out, out)


@libentry()
@triton.jit
def l1_norm_rows_kernel_1(
    X,
    Mid,
    M,
    N,
    MID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    pid = ext.program_id(0).to(tl.int64)
    row = pid // MID_SIZE
    chunk = pid % MID_SIZE
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        X + row * N + offsets,
        mask=(row < M) & (offsets < N),
        other=0.0,
    ).to(tl.float32)
    tl.store(Mid + row * MID_SIZE + chunk, tl.sum(tl.abs(values)), mask=row < M)


@libentry()
@triton.jit
def l1_norm_rows_tail_kernel(
    X,
    Mid,
    M,
    N,
    MID_SIZE: tl.constexpr,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    row = ext.program_id(0).to(tl.int64)
    total = 0.0
    for offset in tl.static_range(TAIL_SIZE):
        value = tl.load(X + row * N + TAIL_OFFSET + offset).to(tl.float32)
        total += tl.abs(value)
    tl.store(Mid + row * MID_SIZE + MID_SIZE - 1, total, mask=row < M)


@libentry()
@triton.jit
def l1_norm_rows_kernel_2(
    Mid,
    Next,
    M,
    MID_SIZE: tl.constexpr,
    NEXT_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    pid = ext.program_id(0).to(tl.int64)
    row = pid // NEXT_SIZE
    chunk = pid % NEXT_SIZE
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    partial = tl.load(
        Mid + row * MID_SIZE + offsets,
        mask=(row < M) & (offsets < MID_SIZE),
        other=0.0,
    ).to(tl.float32)
    tl.store(Next + row * NEXT_SIZE + chunk, tl.sum(partial), mask=row < M)


@libentry()
@triton.jit
def l1_norm_rows_reduce_tail_kernel(
    Mid,
    Next,
    M,
    MID_SIZE: tl.constexpr,
    NEXT_SIZE: tl.constexpr,
    TAIL_OFFSET: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    row = ext.program_id(0).to(tl.int64)
    total = 0.0
    for offset in tl.static_range(TAIL_SIZE):
        total += tl.load(Mid + row * MID_SIZE + TAIL_OFFSET + offset).to(tl.float32)
    tl.store(Next + row * NEXT_SIZE + NEXT_SIZE - 1, total, mask=row < M)


@libentry()
@triton.jit
def l1_norm_rows_kernel_3(
    Next,
    Out,
    M,
    NEXT_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    row = ext.program_id(0).to(tl.int64)
    total = 0.0
    for offset in tl.static_range(NEXT_SIZE):
        total += tl.load(Next + row * NEXT_SIZE + offset).to(tl.float32)
    tl.store(Out + row, total, mask=row < M)


# ---------------------------------------------------------------------------
# Flat L2-norm fast path (ord == 2, dim == all dims; the entry the official
# norm / linalg_vector_norm benchmarks hit).
#
# The legacy path launches one tl.sum program per MID_SIZE chunk plus
# zero-workspace + masked tail + final kernels. This path keeps the same
# proven 1D chunk-reduce (each program reduces exactly one chunk of
# _L2_BLOCK lanes with a single tl.sum -- 8192 plain, the documented
# XPU-safe lane count without buffer_size_limit) but removes every masked
# load (chunks are only launched for exact multiples of the block; the
# non-divisible remainder is accumulated by an exact unmasked tail kernel)
# and drops the zero-workspace pad. All accumulation is fp32; the sqrt
# result is cast to x.dtype on store. NaN and +-inf propagate like the
# legacy sum-of-squares path (NaN stays NaN; inf -> inf), which matches
# torch.norm on CPU and the device for the finite matrices exercised by
# the accuracy tests.
_L2_BLOCK = 8192      # tl.sum-safe plain lane count (stage 2+)
_L2_BLOCK_WIDE = 32768  # stage-1 blocks for big tensors (needs buffer_size_limit=2048)
_L2_SMALL_LIMIT = _L2_BLOCK
_L2_VEC = 512           # lanes per loop iteration in the small/tail/final kernels


def _l2_pick(m, cap=8192, floor=2):
    """Pick the VEC lane count for a chunked accumulate kernel over exactly m
    lanes. The XPU backend only accepts ONE tl.sum width per kernel (mixed
    widths fail in TritonXPULegalize) and scalar loop iterations are ~0.4us
    each, so the sweet spot minimizes iters = m // VEC + m % VEC over
    VEC in pow2 [floor, cap]; the caller then launches one kernel whose
    scalar remainder is <= floor - 1 lanes."""
    best, best_score = floor, 1 << 62
    v = floor
    while v <= cap:
        score = m // v + m % v
        if score < best_score:
            best, best_score = v, score
        v <<= 1
    return best


def _l2_pick2(m, p, cap=8192, floor=2):
    """VEC for the merged tail+final kernel: minimize total iterations over
    m tail lanes (fp32 accumulator) plus p partial lanes (fp64 accumulator)."""
    best, best_score = floor, 1 << 62
    v = floor
    while v <= cap:
        score = m // v + m % v + p // v + p % v
        if score < best_score:
            best, best_score = v, score
        v <<= 1
    return best


# Only merge the stage-1 tail with the final kernel when the number of earlier
# partials is small; otherwise the merged kernel's per-partial fp64 scalar
# loops dominate and the split path stays faster (measured on XPU).
_L2_MERGE_PREV_MAX = 8


@libentry()
@triton.jit
def _l2_flat_small_kernel(X, Out, N, VEC: tl.constexpr):
    """Exact unmasked sum-of-squares over 0 < N <= _L2_SMALL_LIMIT: VEC-wide
    chunks plus a scalar tail for the remainder (two dynamic loops; the
    three-loop and static-unroll variants fail to compile on this XPU)."""
    total = tl.zeros((), dtype=tl.float32)
    full = (N // VEC) * VEC
    for off in tl.range(0, full, VEC):
        v = tl.load(X + off + tl.arange(0, VEC)).to(tl.float32)
        total += tl.sum(v * v)
    for off in tl.range(0, N - full):
        v = tl.load(X + full + off).to(tl.float32)
        total += v * v
    tl.store(Out, tl.sqrt(total))


@libentry()
@triton.jit
def _l2_flat_tail_kernel(X, Out, TAIL_N, SQUARED: tl.constexpr,
                         VEC: tl.constexpr):
    """Exact unmasked chunk accumulation (squares when SQUARED); callers pass
    a VIEW starting at the tail so the pointer stays affine; VEC chunks plus
    a scalar tail for the remainder (the compilable two-loop shape)."""
    total = tl.zeros((), dtype=tl.float32)
    full = (TAIL_N // VEC) * VEC
    for off in tl.range(0, full, VEC):
        v = tl.load(X + off + tl.arange(0, VEC)).to(tl.float32)
        if SQUARED:
            v = v * v
        total += tl.sum(v)
    for off in tl.range(0, TAIL_N - full):
        v = tl.load(X + full + off).to(tl.float32)
        if SQUARED:
            v = v * v
        total += v
    tl.store(Out, total)


@libentry()
@triton.jit
def _l2_flat_blk_kernel(
    X,
    Partial,
    BLOCK: tl.constexpr,
    SQUARED: tl.constexpr,
    buffer_size_limit: tl.constexpr,
):
    """One fp32 partial per contiguous BLOCK-lane chunk of the flat input.
    Masks-free: every lane is in-bounds, the pointer stays affine
    (X + pid * BLOCK + arange), and the tl.sum lane count is a safe size
    (8192 plain / 32768 w/ bsl=2048)."""
    pid = ext.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + offs).to(tl.float32)
    if SQUARED:
        v = v * v
    tl.store(Partial + pid, tl.sum(v))


@libentry()
@triton.jit
def _l2_flat_final_kernel(Partial, Out, N, VEC: tl.constexpr):
    """Sum <= 8192 fp32 partials in fp64 and sqrt into Out. The fp64
    accumulation keeps the final rounding far below one output ulp for all
    supported input dtypes, matching torch's result to within the rounding
    of the output dtype (the fp32 mid-array rounding is ~1e-7 relative)."""
    total = tl.zeros((), dtype=tl.float64)
    full = (N // VEC) * VEC
    for off in tl.range(0, full, VEC):
        total += tl.sum(
            tl.load(Partial + off + tl.arange(0, VEC)).to(tl.float64)
        )
    for off in tl.range(0, N - full):
        total += tl.load(Partial + full + off).to(tl.float64)
    tl.store(Out, tl.sqrt(total))


@libentry()
@triton.jit
def _l2_flat_merge_tail_final_kernel(
    X,
    Partial,
    Out,
    TAIL_N,
    MID_PREV: tl.constexpr,
    SQUARED: tl.constexpr,
    VEC: tl.constexpr,
):
    """Fused stage-1 tail + final accumulate: adds the fp32 tail sum to the
    earlier fp32 partials (fp64), then sqrt into Out, in ONE launch. Exact
    and unmasked: the tail is processed in VEC-wide chunks plus the scalar
    remainder; both loops use the single VEC tl.sum width the XPU backend
    accepts. Only used when MID_PREV (partial count) is small."""
    total = tl.zeros((), dtype=tl.float32)
    full = (TAIL_N // VEC) * VEC
    for off in tl.range(0, full, VEC):
        v = tl.load(X + off + tl.arange(0, VEC)).to(tl.float32)
        if SQUARED:
            v = v * v
        total += tl.sum(v)
    for off in tl.range(0, TAIL_N - full):
        v = tl.load(X + full + off).to(tl.float32)
        if SQUARED:
            v = v * v
        total += v
    ft = total.to(tl.float64)
    fullp = (MID_PREV // VEC) * VEC
    for off in tl.range(0, fullp, VEC):
        ft += tl.sum(tl.load(Partial + off + tl.arange(0, VEC)).to(tl.float64))
    for off in tl.range(0, MID_PREV - fullp):
        ft += tl.load(Partial + fullp + off).to(tl.float64)
    tl.store(Out, tl.sqrt(ft))


def _l2_flat(x, out):
    """Compute ||x||_2 over the whole flat contiguous x into scalar out."""
    n = x.numel()
    x1 = x if x.dim() == 1 else x.view(-1)
    if n <= _L2_SMALL_LIMIT:
        _l2_flat_small_kernel[(1,)](x1, out, n, VEC=_l2_pick(n))
        return

    # Stage 1: wide (32768-wide) blocks; the residue is split into 8192-lane
    # chunks plus one exact < 8192-lane tail (views, affine pointers).
    rows1 = n // _L2_BLOCK_WIDE
    tail1 = n - rows1 * _L2_BLOCK_WIDE
    t_rows = tail1 // _L2_BLOCK
    t_rem = tail1 - t_rows * _L2_BLOCK
    prev = rows1 + t_rows
    mid_cnt = prev + (1 if t_rem else 0)
    mid = torch.empty((mid_cnt,), dtype=torch.float32, device=x.device)
    if rows1:
        _l2_flat_blk_kernel[(rows1,)](
            x1, mid, _L2_BLOCK_WIDE, SQUARED=True, buffer_size_limit=2048
        )
    if t_rows:
        _l2_flat_blk_kernel[(t_rows,)](
            x1[rows1 * _L2_BLOCK_WIDE :],
            mid[rows1:],
            _L2_BLOCK,
            SQUARED=True,
            buffer_size_limit=2048,
        )
    if t_rem:
        if prev <= _L2_MERGE_PREV_MAX:
            # Fuse tail + final into one launch: saves the separate tail and
            # final kernels (plus their launch latencies) for small partial
            # counts; exact fp32 tail + fp64 partial accumulation preserves the
            # split-path numerics.
            _l2_flat_merge_tail_final_kernel[(1,)](
                x1[rows1 * _L2_BLOCK_WIDE + t_rows * _L2_BLOCK :],
                mid,
                out,
                t_rem,
                MID_PREV=prev,
                SQUARED=True,
                VEC=_l2_pick2(t_rem, prev),
            )
            return
        _l2_flat_tail_kernel[(1,)](
            x1[rows1 * _L2_BLOCK_WIDE + t_rows * _L2_BLOCK :],
            mid[rows1 + t_rows :],
            t_rem,
            SQUARED=True,
            VEC=_l2_pick(t_rem),
        )

    # Stages 2..: reduce partials with 8192-lane blocks until a scalar.
    while mid_cnt > _L2_BLOCK:
        rows2 = mid_cnt // _L2_BLOCK
        rem2 = mid_cnt - rows2 * _L2_BLOCK
        next_cnt = rows2 + (1 if rem2 else 0)
        nxt = torch.empty((next_cnt,), dtype=torch.float32, device=x.device)
        if rows2:
            _l2_flat_blk_kernel[(rows2,)](
                mid, nxt, _L2_BLOCK, SQUARED=False, buffer_size_limit=2048
            )
        if rem2:
            _l2_flat_tail_kernel[(1,)](
                mid[rows2 * _L2_BLOCK :],
                nxt[rows2:],
                rem2,
                SQUARED=False,
                VEC=_l2_pick(rem2),
            )
        mid = nxt
        mid_cnt = next_cnt

    _l2_flat_final_kernel[(1,)](mid, out, mid_cnt, VEC=_l2_pick(mid_cnt))


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS_KUNLUNXIN VECTOR_NORM")
    if dtype is None:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    if dim is None:
        dim = list(range(x.ndim))
    elif isinstance(dim, int):
        dim = [dim]
    else:
        dim = list(dim)
    normalized_dim = []
    for d in dim:
        if d < -x.ndim or d >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-x.ndim}, {x.ndim - 1}], but got {d})"
            )
        normalized_dim.append(d % x.ndim)
    if len(set(normalized_dim)) != len(normalized_dim):
        raise RuntimeError("dim must contain all distinct dimensions")
    dim = normalized_dim

    with torch_device_fn.device(x.device):
        if len(dim) == x.ndim:
            shape = [1] * x.ndim
            x = dim_compress(x, dim)
            M = x.numel()
            # Fast flat L2 (ord == 2, all dims, non-empty): mask-free 2D row
            # reduce + exact tail, avoiding the legacy multi-kernel path below.
            # Empty tensors (M == 0) keep the legacy zero-sum semantics.
            if ord == 2 and M > 0:
                out = torch.empty(shape, dtype=dtype, device=x.device)
                _l2_flat(x, out)
                if not keepdim:
                    out = out.squeeze(dim=dim)
                return out
            cluster_num = 12
            # XPU: tl.sum over a 1D tile is only correct up to a bounded lane
            # count. Empirically BLOCK_SIZE=32768 (bsl=2048) is the safe max;
            # a larger tile silently drops lanes. Cap here (dtype-independent)
            # keeps stage-1 tiles correct AND bounds MID_SIZE <= 32768 for all
            # M <= 2**30 so stage-2's tl.sum(mid) is also within the safe range.
            # The old cap int(1024*64/element_size) gave 16384 for fp32 -> for
            # M=2**30 MID_SIZE=65536 which broke stage-2 (wrong fp32 results).
            BLOCK_SIZE = min(
                triton.next_power_of_2(triton.cdiv(M, cluster_num)),
                32768,
            )
            MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            # Stage-2 reduces a power-of-two tile. Pad and explicitly clear its
            # workspace so XPU masked loads never consume memory past MID_SIZE.
            mid = torch.empty([BLOCK_MID], dtype=torch.float32, device=x.device)
            zero_workspace_kernel[(1,)](mid, BLOCK_MID)
            out = torch.empty(shape, dtype=dtype, device=x.device)
            if ord == 2:
                l2_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                tail_size = M % BLOCK_SIZE
                if tail_size:
                    l2_norm_tail_kernel[(1,)](
                        x,
                        mid,
                        M - tail_size,
                        tail_size,
                        MID_SIZE - 1,
                        buffer_size_limit=2048,
                    )
                l2_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == float("inf"):
                max_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                max_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == -float("inf"):
                min_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                min_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            elif ord == 0:
                l0_norm_kernel_1[(MID_SIZE,)](
                    x, mid, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                tail_size = M % BLOCK_SIZE
                if tail_size:
                    l0_norm_tail_kernel[(1,)](
                        x,
                        mid,
                        M - tail_size,
                        tail_size,
                        MID_SIZE - 1,
                        buffer_size_limit=2048,
                    )
                l0_norm_kernel_2[(1,)](
                    mid, out, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
            else:
                l1_norm_kernel_1[(MID_SIZE,)](
                    x, mid, ord, M, BLOCK_SIZE, buffer_size_limit=2048
                )
                tail_size = M % BLOCK_SIZE
                if tail_size:
                    l1_norm_tail_kernel[(1,)](
                        x,
                        mid,
                        ord,
                        M - tail_size,
                        tail_size,
                        MID_SIZE - 1,
                        buffer_size_limit=2048,
                    )
                l1_norm_kernel_2[(1,)](
                    mid, out, ord, MID_SIZE, BLOCK_MID, buffer_size_limit=2048
                )
        else:
            shape = list(x.shape)
            dim = [d % x.ndim for d in dim]
            x = dim_compress(x, dim)
            N = 1
            for i in dim:
                N *= shape[i]
                shape[i] = 1
            M = x.numel() // N
            out = torch.empty(shape, dtype=dtype, device=x.device)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
            if ord == 2:
                l2_norm_kernel[grid](x, out, M, N)
            elif ord == float("inf"):
                max_norm_kernel[grid](x, out, M, N)
            elif ord == -float("inf"):
                min_norm_kernel[grid](x, out, M, N)
            elif ord == 0:
                l0_norm_kernel[grid](x, out, M, N)
            elif ord == 1 and N > 1024:
                BLOCK_SIZE = 1024
                MID_SIZE = triton.cdiv(N, BLOCK_SIZE)
                mid = torch.empty((M, MID_SIZE), dtype=torch.float32, device=x.device)
                l1_norm_rows_kernel_1[(M * MID_SIZE,)](
                    x,
                    mid,
                    M,
                    N,
                    MID_SIZE,
                    BLOCK_SIZE,
                    buffer_size_limit=2048,
                )
                tail_size = N % BLOCK_SIZE
                if tail_size:
                    l1_norm_rows_tail_kernel[(M,)](
                        x,
                        mid,
                        M,
                        N,
                        MID_SIZE,
                        N - tail_size,
                        tail_size,
                        buffer_size_limit=2048,
                    )
                if MID_SIZE <= 1024:
                    l1_norm_rows_kernel_3[(M,)](
                        mid,
                        out,
                        M,
                        MID_SIZE,
                        triton.next_power_of_2(MID_SIZE),
                        buffer_size_limit=2048,
                    )
                else:
                    NEXT_SIZE = triton.cdiv(MID_SIZE, BLOCK_SIZE)
                    BLOCK_NEXT = triton.next_power_of_2(NEXT_SIZE)
                    next_mid = torch.empty(
                        (M, NEXT_SIZE), dtype=torch.float32, device=x.device
                    )
                    l1_norm_rows_kernel_2[(M * NEXT_SIZE,)](
                        mid,
                        next_mid,
                        M,
                        MID_SIZE,
                        NEXT_SIZE,
                        BLOCK_SIZE,
                        buffer_size_limit=2048,
                    )
                    tail_size = MID_SIZE % BLOCK_SIZE
                    if tail_size:
                        l1_norm_rows_reduce_tail_kernel[(M,)](
                            mid,
                            next_mid,
                            M,
                            MID_SIZE,
                            NEXT_SIZE,
                            MID_SIZE - tail_size,
                            tail_size,
                            buffer_size_limit=2048,
                        )
                    l1_norm_rows_kernel_3[(M,)](
                        next_mid,
                        out,
                        M,
                        NEXT_SIZE,
                        BLOCK_NEXT,
                        buffer_size_limit=2048,
                    )
            else:
                v_norm_kernel[grid](x, out, M, N, ord, isCloseUnrollControl=True)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
