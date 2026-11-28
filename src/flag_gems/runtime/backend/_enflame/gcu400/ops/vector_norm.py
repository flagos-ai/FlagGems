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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim

pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

_min = builtins.min
_max = builtins.max
MAX_GRID = 48
# Grid cap for the dim path (ord=2 row-accumulator kernel): the kernel uses tl.range with a persistent grid,
# and raising the cap to 384 gives up to 20% more bandwidth utilization for large-M cases (e.g. 524288x512).
DIM_GRID_CAP = 384
_MID_WORKSPACES = {}


# ============================================================
# Dim-reduction kernels (reduce along last dim after dim_compress)
# ============================================================


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def l2_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    rows_in_block = tl.arange(0, BLOCK_M)
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        row_index = tile_id * BLOCK_M + rows_in_block
        row_mask = row_index < M

        # Immediately reduce each column block into a [BLOCK_M] row accumulator: the old form kept an [BLOCK_M, BLOCK_N]
        # fp32 accumulator (BLOCK_M=64/BLOCK_N=512, i.e. 32768 fp32 values), causing severe
        # register spilling, the root cause of the dim path being slower than torch.
        row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        for col_offset in range(0, N, BLOCK_N):
            col_index = col_offset + tl.arange(0, BLOCK_N)
            mask = row_mask[:, None] & (col_index < N)[None, :]
            x_tile = tl.load(
                X + row_index[:, None] * N + col_index[None, :], mask, other=0.0
            ).to(tl.float32)
            row_sum += tl.sum(x_tile * x_tile, axis=1)

        tl.store(Out + row_index, tl.sqrt(row_sum), row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def max_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=0.0).to(tl.float32)
            _max = tl.maximum(tl.abs(a), _max)

        result = tl.max(_max, axis=1)[:, None]
        tl.store(Out + m_offset, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def min_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=float("inf")).to(
                tl.float32
            )
            _min = tl.minimum(tl.abs(a), _min)

        result = tl.min(_min, axis=1)[:, None]
        tl.store(Out + m_offset, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def l0_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=0).to(tl.float32)
            _sum += tl.where(a != 0, 1.0, 0.0)

        result = tl.sum(_sum, axis=1)[:, None]
        tl.store(Out + m_offset, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def l1_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            a = tl.load(X + m_offset * N + cols, mask, other=0.0).to(tl.float32)
            _sum += tl.abs(a)

        result = tl.sum(_sum, axis=1)[:, None]
        tl.store(Out + m_offset, result, row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "ord"])
def v_norm_dim_kernel(
    X,
    Out,
    M,
    N,
    ord,
    ORD_NEG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            mask = (m_offset < M) & (cols < N)
            # Masked lanes must contribute 0. pow(0, ord) is +inf for a
            # negative order (aten issue #4600), so for ORD_NEG the filler is
            # +inf instead: pow(|inf|, negative ord) == 0. A tl.where around
            # the pow miscompiles to inf on 2D tiles on this backend, so the
            # guard is expressed purely through the load filler value.
            if ORD_NEG:
                a = tl.load(X + m_offset * N + cols, mask, other=float("inf")).to(
                    tl.float32
                )
            else:
                a = tl.load(X + m_offset * N + cols, mask, other=0.0).to(tl.float32)
            _sum += pow(tl.abs(a), ord)

        total = tl.sum(_sum, axis=1)
        result = pow(total, 1.0 / ord)[:, None]
        tl.store(Out + m_offset, result, row_mask)


# ============================================================
# Global single-pass kernels (loop-based, 1 launch for small/medium M)
# ============================================================


@libentry()
@triton.jit(do_not_specialize=["M"])
def l2_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    tl.store(Out, tl.sqrt(tl.sum(acc)))


@libentry()
@triton.jit(do_not_specialize=["M"])
def max_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc = tl.maximum(tl.abs(x), acc)
    tl.store(Out, tl.max(acc))


@libentry()
@triton.jit(do_not_specialize=["M"])
def min_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.full([BLOCK], value=float("inf"), dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=float("inf")).to(tl.float32)
        acc = tl.minimum(tl.abs(x), acc)
    tl.store(Out, tl.min(acc))


@libentry()
@triton.jit(do_not_specialize=["M"])
def l0_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += (x != 0).to(tl.float32)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit(do_not_specialize=["M"])
def l1_global_single(X, Out, M, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += tl.abs(x)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit(do_not_specialize=["M", "ord"])
def v_global_single(X, Out, ord, M, ORD_NEG: tl.constexpr, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        offset = off + tl.arange(0, BLOCK)
        mask = offset < M
        # Masked lanes must contribute 0; for a negative order the filler is
        # +inf since pow(|inf|, ord) == 0 (aten issue #4600 - pow(0, ord)
        # would be +inf and poison the sum).
        if ORD_NEG:
            x = tl.load(X + offset, mask=mask, other=float("inf")).to(tl.float32)
        else:
            x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        acc += pow(tl.abs(x), ord)
    tl.store(Out, pow(tl.sum(acc), 1.0 / ord))


# ============================================================
# Global-reduction kernels (two-stage: partial -> final)
#
# GCU400 tuning notes (validated by probes, see case log 2026-11-16):
#   * partial kernel keeps a [BLOCK]-lane fp32 accumulator in the hot loop
#     (pure elementwise FMA, zero cross-lane traffic) and runs exactly one
#     reduction tree per program at the very end. The single masked tail
#     block is folded in by its owner program, keeping the hot loop free of
#     any mask predicate. A per-chunk tree (the previous scheme) measured
#     ~15% slower on 0.5G-element tensors.
#   * the BLOCK=65536 lane accumulator does not fit the 917504B DSM budget
#     (required 919552B), so 16-bit inputs cap at BLOCK=16384 and fp32 caps
#     at BLOCK=32768.
#   * 16-bit loads move ~40GB/s/program vs ~58GB/s/program for fp32 lanes:
#     packing two fp16/bf16 elements per int32 lane (vnorm_global_pack16)
#     restores the fp32 byte rate on large tensors (1.96 -> 2.54 TB/s on
#     the 0.5G-element core shape, numerically bit-exact including
#     subnormals / +-inf / NaN payloads).
#   * the merge buffer only holds one partial per program (grid <= 48), so
#     the final kernel is a single narrow tree over <= 64 lanes.
# ============================================================


@libentry()
@triton.jit(do_not_specialize=["M", "ord"])
def vnorm_global_partial(
    X,
    Mid,
    M,
    ord,
    MODE: tl.constexpr,
    ORD_NEG: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    # MODE: 0 = ord 2 (squares), 1 = ord 1 (|x|), 2 = ord inf (max |x|),
    #       3 = ord -inf (min |x|), 4 = ord 0 (nonzero count), 5 = generic
    #       pow(|x|, ord). ORD_NEG guards the masked tail only: pow(0, ord)
    #   with a negative order would poison the sum with +inf.
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_full = M // BLOCK
    if MODE == 3:
        acc = tl.full([BLOCK], value=float("inf"), dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
    for chunk in tl.range(pid, num_full, num_prog):
        offset = chunk * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(X + offset).to(tl.float32)
        if MODE == 0:
            acc += a * a
        elif MODE == 1:
            acc += tl.abs(a)
        elif MODE == 2:
            acc = tl.maximum(acc, tl.abs(a))
        elif MODE == 3:
            acc = tl.minimum(acc, tl.abs(a))
        elif MODE == 4:
            acc += (a != 0).to(tl.float32)
        else:
            acc += pow(tl.abs(a), ord)
    if HAS_TAIL:
        if pid == num_full % num_prog:
            offset = num_full * BLOCK + tl.arange(0, BLOCK)
            mask = offset < M
            # Tail filler by MODE: min wants +inf (identity for minimum), the
            # generic pow mode wants +inf for a negative order because
            # pow(0, ord) is +inf (aten issue #4600) while pow(|inf|, ord)
            # is 0; every other mode treats 0.0 as the exact identity.
            if MODE == 3 or (MODE == 5 and ORD_NEG):
                a = tl.load(X + offset, mask=mask, other=float("inf")).to(tl.float32)
            else:
                a = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
            if MODE == 0:
                acc += a * a
            elif MODE == 1:
                acc += tl.abs(a)
            elif MODE == 2:
                acc = tl.maximum(acc, tl.abs(a))
            elif MODE == 3:
                acc = tl.minimum(acc, tl.abs(a))
            elif MODE == 4:
                acc += (a != 0).to(tl.float32)
            else:
                acc += pow(tl.abs(a), ord)
    if MODE == 2:
        tl.store(Mid + pid, tl.max(acc))
    elif MODE == 3:
        tl.store(Mid + pid, tl.min(acc))
    else:
        tl.store(Mid + pid, tl.sum(acc))


@libentry()
@triton.jit(do_not_specialize=["M"])
def vnorm_global_pack16(
    X,
    Mid,
    M,
    IS_BF16: tl.constexpr,
    BLOCK32: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    # Ord-2 fast path for fp16/bf16 inputs with an even element count: X is a
    # zero-copy int32 view of the 16-bit buffer, so each lane carries two
    # elements and the load path recovers fp32-lane byte efficiency. The
    # bitcast reconstruction is exact (subnormals, +-inf and NaN payloads
    # included); squares are accumulated in fp32 like the plain path. The hot
    # loop walks a block pointer (strided per program, no boundary_check -
    # full chunks are exact by construction), which measured 3% faster than
    # the arange form on 0.5G-element tensors; the single masked word tail is
    # folded in by its owner on the plain path.
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_words = M // 2
    num_full = num_words // BLOCK32
    acc = tl.zeros([BLOCK32], dtype=tl.float32)
    if pid < num_full:
        ptr = tl.make_block_ptr(
            base=X,
            shape=(num_full * BLOCK32,),
            strides=(1,),
            offsets=(pid * BLOCK32,),
            block_shape=(BLOCK32,),
            order=(0,),
        )
        for _chunk in tl.range(pid, num_full, num_prog):
            v = tl.load(ptr)
            lo = (v & 0xFFFF).to(tl.uint16)
            hi = ((v >> 16) & 0xFFFF).to(tl.uint16)
            if IS_BF16:
                a = lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
                b = hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
            else:
                a = lo.to(tl.float16, bitcast=True).to(tl.float32)
                b = hi.to(tl.float16, bitcast=True).to(tl.float32)
            acc += a * a
            acc += b * b
            ptr = tl.advance(ptr, (num_prog * BLOCK32,))
    if HAS_TAIL:
        if pid == num_full % num_prog:
            offset = num_full * BLOCK32 + tl.arange(0, BLOCK32)
            mask = offset < num_words
            v = tl.load(X + offset, mask=mask, other=0)
            lo = (v & 0xFFFF).to(tl.uint16)
            hi = ((v >> 16) & 0xFFFF).to(tl.uint16)
            if IS_BF16:
                a = lo.to(tl.bfloat16, bitcast=True).to(tl.float32)
                b = hi.to(tl.bfloat16, bitcast=True).to(tl.float32)
            else:
                a = lo.to(tl.float16, bitcast=True).to(tl.float32)
                b = hi.to(tl.float16, bitcast=True).to(tl.float32)
            # other=0 unpacks to 0.0 which is the exact identity for squares
            acc += a * a + b * b
    tl.store(Mid + pid, tl.sum(acc))


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE", "ord"])
def vnorm_global_final(
    Mid,
    Out,
    ord,
    MID_SIZE,
    MODE: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < MID_SIZE
    if MODE == 2:
        mid = tl.load(Mid + offsets, mask=mask, other=0.0)
        tl.store(Out, tl.max(mid))
    elif MODE == 3:
        mid = tl.load(Mid + offsets, mask=mask, other=float("inf"))
        tl.store(Out, tl.min(mid))
    else:
        mid = tl.load(Mid + offsets, mask=mask, other=0.0)
        total = tl.sum(mid)
        if MODE == 0:
            tl.store(Out, tl.sqrt(total))
        elif MODE == 1:
            tl.store(Out, total)
        elif MODE == 4:
            tl.store(Out, total)
        else:
            tl.store(Out, pow(total, 1.0 / ord))


def _ord_mode(ord_val):
    if ord_val == 2:
        return 0
    if ord_val == 1:
        return 1
    if ord_val == float("inf"):
        return 2
    if ord_val == -float("inf"):
        return 3
    if ord_val == 0:
        return 4
    return 5


def _mid_workspace(device, size):
    """Reusable staging buffer for the two-pass reduction.

    Allocating ``torch.empty([grid])`` per reduction costs 2.1us of host time, which is about
    15% of the total at shapes like M~1M; the buffer only holds temporary accumulators, so
    caching it by (device, size) is safe since reuse on one stream is naturally ordered.
    """
    key = (device.type, device.index, size)
    buf = _MID_WORKSPACES.get(key)
    if buf is None:
        buf = torch.empty([size], dtype=torch.float32, device=device)
        if len(_MID_WORKSPACES) < 64:
            _MID_WORKSPACES[key] = buf
    return buf


def _launch_global_norm(x, out, M, ord_val, dtype):
    SINGLE_PASS_LIMIT = 65536
    SINGLE_BLOCK = 32768
    PACK16_MIN_NUMEL = 1 << 23
    PACK16_BLOCK32 = 32768
    ord_neg = isinstance(ord_val, (int, float)) and ord_val < 0

    with torch_device_fn.device(x.device):
        if M <= SINGLE_PASS_LIMIT:
            bs = _max(1024, _min(triton.next_power_of_2(M), SINGLE_BLOCK))
            if ord_val == 2:
                l2_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            elif ord_val == float("inf"):
                max_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            elif ord_val == -float("inf"):
                min_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            elif ord_val == 0:
                l0_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            elif ord_val == 1:
                l1_global_single[(1,)](x, out, M, BLOCK=bs, num_warps=1)
            else:
                v_global_single[(1,)](
                    x, out, ord_val, M, ORD_NEG=ord_neg, BLOCK=bs, num_warps=1
                )
            return

        # Ord-2 packed fast path for 16-bit inputs: two elements per int32
        # lane restore fp32-lane load efficiency on large tensors.
        if (
            x.dtype in (torch.float16, torch.bfloat16)
            and ord_val == 2
            and M % 2 == 0
            and M >= PACK16_MIN_NUMEL
        ):
            num_words = M // 2
            num_chunks = triton.cdiv(num_words, PACK16_BLOCK32)
            grid = _min(num_chunks, MAX_GRID)
            mid = _mid_workspace(x.device, grid)
            # The packed path requires an even last dimension; otherwise view(int32) errors out directly (e.g.
            # a contiguous tensor with an odd last dimension such as (200, 40999, 3)). Flatten to 1D first and then take
            # an
            # int32 view; the prerequisite that the total element count is even is guaranteed by M % 2 == 0 below.
            vnorm_global_pack16[(grid,)](
                x.reshape(-1).view(torch.int32),
                mid,
                M,
                IS_BF16=(x.dtype == torch.bfloat16),
                BLOCK32=PACK16_BLOCK32,
                HAS_TAIL=(num_words % PACK16_BLOCK32) != 0,
                num_warps=8,
            )
            vnorm_global_final[(1,)](
                mid,
                out,
                ord_val,
                grid,
                MODE=0,
                BLOCK_MID=triton.next_power_of_2(grid),
                num_warps=1,
            )
            return

        mode = _ord_mode(ord_val)
        # fp32 lanes move ~45% more bytes per program than 16-bit lanes, so
        # fp32 takes the widest block its DSM budget allows; 16-bit stays at
        # 16384 lanes. Small fp32 tensors prefer more parallelism over width.
        if x.element_size() >= 4:
            if M <= (1 << 21):
                block, num_warps = 8192, 4
            else:
                block, num_warps = 32768, 2
        else:
            block, num_warps = 16384, 4
        num_chunks = triton.cdiv(M, block)
        grid = _min(num_chunks, MAX_GRID)

        mid = _mid_workspace(x.device, grid)
        vnorm_global_partial[(grid,)](
            x,
            mid,
            M,
            ord_val,
            MODE=mode,
            ORD_NEG=ord_neg,
            BLOCK=block,
            HAS_TAIL=(M % block) != 0,
            num_warps=num_warps,
        )
        vnorm_global_final[(1,)](
            mid,
            out,
            ord_val,
            grid,
            MODE=mode,
            BLOCK_MID=triton.next_power_of_2(grid),
            num_warps=1,
        )


def _launch_dim_norm(x, out, M, N, ord_val):
    BLOCK_N = _min(triton.next_power_of_2(N), 2048)
    BLOCK_M = _max(1, _min(128, 32768 // BLOCK_N))
    grid_m = _min(triton.cdiv(M, BLOCK_M), MAX_GRID)

    # ord=2 uses the row-accumulator kernel, whose register usage scales only linearly with BLOCK_M, so it can (and
    # should)
    # use a much larger BLOCK_M and a wider grid than before. Binning is based on measurements (GCU400):
    #   For N<=512, 256 rows per program is most cost-effective (32768x512: 47.6µs→32.2µs);
    #   for N>=1024, 64 rows is optimal (4096x4096: 47.9µs→29.0µs), and larger values regress;
    #   with N>=4096 and fp32, BLOCK_M*BLOCK_N above 131072 elements causes OutOfResources directly.
    # For very small M, fall back to about 8 rows per program to avoid an overly small grid. num_warps is uniformly 2:
    # measurements show that
    # reducing a small BLOCK_M to 1 warp regresses fp32 [64,64] from 5.2µs to 10.9µs.
    if ord_val == 2:
        l2_block_m = 256 if BLOCK_N <= 512 else 64
        l2_block_m = _min(l2_block_m, _max(8, triton.next_power_of_2(_max(1, M // 8))))
        l2_grid = _min(triton.cdiv(M, l2_block_m), DIM_GRID_CAP)
        with torch_device_fn.device(x.device):
            l2_norm_dim_kernel[(l2_grid,)](
                x,
                out,
                M,
                N,
                BLOCK_M=l2_block_m,
                BLOCK_N=BLOCK_N,
                num_warps=2,
            )
        return

    with torch_device_fn.device(x.device):
        if ord_val == float("inf"):
            max_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        elif ord_val == -float("inf"):
            min_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        elif ord_val == 0:
            l0_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        elif ord_val == 1:
            l1_norm_dim_kernel[(grid_m,)](
                x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=1
            )
        else:
            v_norm_dim_kernel[(grid_m,)](
                x,
                out,
                M,
                N,
                ord_val,
                ORD_NEG=isinstance(ord_val, (int, float)) and ord_val < 0,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=1,
            )


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logger.debug("GEMS_ENFLAME VECTOR_NORM")
    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        elif not isinstance(dtype, torch.dtype):
            dtype = torch.float32
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    if (not dim) or len(dim) == x.ndim:
        # Global reduction depends only on a flat memory view: contiguous inputs need no dim_compress
        # permute + contiguous. With keepdim=False a 0-dim output is built directly,
        # also saving squeeze (~6us host overhead on small/medium shapes).
        if x.is_contiguous():
            inp = x
        else:
            inp = dim_compress(x, list(range(x.ndim)))
        M = inp.numel()
        if keepdim:
            out = torch.empty([1] * x.ndim, dtype=dtype, device=x.device)
            _launch_global_norm(inp, out, M, ord, dtype)
            return out
        out = torch.empty((), dtype=dtype, device=x.device)
        _launch_global_norm(inp, out, M, ord, dtype)
        return out
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
        _launch_dim_norm(x, out, M, N, ord)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
