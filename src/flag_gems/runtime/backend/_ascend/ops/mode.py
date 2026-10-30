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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.mode import _mode_byte, _mode_sort
from flag_gems.ops.sort import convert_to_uint_preverse_order
from flag_gems.ops.topk import _get_iinfo_val
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils import CORE_NUM

logger = logging.getLogger(__name__)
ModeOut = namedtuple("mode", ["values", "indices"])


@libentry()
@triton.jit
def _mode_fused(X, V, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    rows_per_core = tl.cdiv(M, tl.num_programs(0))
    begin = tl.program_id(0) * rows_per_core
    for row in range(begin, tl.minimum(begin + rows_per_core, M)):
        c = tl.arange(0, B)
        if X.dtype.element_ty.is_floating():
            limit = float("inf")
        else:
            limit = _get_iinfo_val(X.dtype.element_ty, return_max=True)
        x = tl.load(X + row * N + c, c < N, other=limit)
        if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
            x = x.to(tl.float32)
        if B == 1:
            ordered = x
        else:
            ordered = tl.sort(x, descending=False)
        ones = tl.full((B,), 1, tl.int32)
        _, count, _ = tl.associative_scan((ordered, ones, ones), 0, _run_count)
        count = tl.where(c < N, count, 0)
        most = tl.max(count, 0)
        at = tl.min(tl.where(count == most, c, B), 0)
        value = tl.sum(tl.where(c == at, ordered, 0), 0)
        tl.store(V + row, value)


@libentry()
@triton.jit
def _mode_find_indices(X, V, IPtr, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    rows_per_core = tl.cdiv(M, tl.num_programs(0))
    begin = tl.program_id(0) * rows_per_core
    c = tl.arange(0, B)
    for row in range(begin, tl.minimum(begin + rows_per_core, M)):
        value = tl.load(V + row)
        x = tl.load(X + row * N + c, c < N, other=0)
        if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
            x = x.to(tl.float32)
            value = value.to(tl.float32)
        matches = (c < N) & ((x == value) | ((x != x) & (value != value)))
        index = tl.min(tl.where(matches, c.to(tl.float32), float(B)), 0).to(tl.int32)
        tl.store(IPtr + row, index)


@triton.jit
def _run_count(av, ac, al, bv, bc, bl):
    # A concatenation extends the left run only if the right segment is uniform.
    count = tl.where((bc == bl) & (av == bv), ac + bc, bc)
    return bv, count, al + bl


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@libentry()
@triton.jit
def _mode_sorted(
    X, IX, V, out_indices, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr
):
    rows_per_core = tl.cdiv(M, tl.num_programs(0))
    begin = tl.program_id(0) * rows_per_core
    for row in range(begin, tl.minimum(begin + rows_per_core, M)):
        c = tl.arange(0, B)
        carry = 0
        best_count = 0
        best_pos = 0
        for base in range(tl.cdiv(N, B)):
            p = base * B + c
            x = tl.load(X + row * N + p, p < N, other=0)
            # Keep predecessor addresses inside the row, including masked lanes.
            prev_p = tl.minimum(tl.maximum(p - 1, 0), N - 1)
            prev = tl.load(X + row * N + prev_p)
            if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
                x = x.to(tl.float32)
                prev = prev.to(tl.float32)
            starts = tl.where((p == 0) | (x != prev), p, carry)
            starts = tl.associative_scan(starts, 0, _maximum)
            count = tl.where(p < N, p - starts + 1, 0)
            most = tl.max(count, 0)
            at = tl.min(tl.where(count == most, p, N), 0)
            better = most > best_count
            best_pos = tl.where(better, at, best_pos)
            best_count = tl.maximum(best_count, most)
            carry = tl.max(tl.where(p < N, starts, 0), 0)
        value = tl.load(X + row * N + best_pos)
        tl.store(V + row, value)
        if IX is not None:
            index = tl.load(IX + row * N + best_pos)
            tl.store(out_indices + row, index)


@libentry()
@triton.jit
def _mode_histogram16(X, H, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    nt = tl.cdiv(N, B)
    per = tl.cdiv(M * nt, tl.num_programs(0))
    c = tl.arange(0, B)
    for tile in range(
        tl.program_id(0) * per, tl.minimum((tl.program_id(0) + 1) * per, M * nt)
    ):
        row = tile // nt
        p = tile % nt * B + c
        x = tl.load(X + row * N + p, p < N, other=0)
        if X.dtype.element_ty.is_floating():
            x = tl.where(x == 0, 0.0, x).to(X.dtype.element_ty)
        # Mask after promotion: uint16 conversion may sign-extend on Ascend.
        key = convert_to_uint_preverse_order(x, False).to(tl.int32) & 65535
        if X.dtype.element_ty.is_floating():
            is_nan = x != x
            tl.atomic_add(H + row * 65536 + key, 1, (p < N) & ~is_nan, sem="relaxed")
            # NaNs do not form equal-value runs in the sorting implementation.
            tl.atomic_max(H + row * 65536 + key, 1, (p < N) & is_nan, sem="relaxed")
        else:
            tl.atomic_add(H + row * 65536 + key, 1, p < N, sem="relaxed")


@libentry()
@triton.jit
def _mode_histogram16_select(H, V, M: tl.constexpr, B: tl.constexpr):
    per = tl.cdiv(M, tl.num_programs(0))
    c = tl.arange(0, B)
    for row in range(
        tl.program_id(0) * per, tl.minimum((tl.program_id(0) + 1) * per, M)
    ):
        best_count = 0
        best_key = 0
        for start in range(0, 65536, B):
            k = start + c
            count = tl.load(H + row * 65536 + k)
            most = tl.max(count, 0)
            local = tl.min(tl.where(count == most, c.to(tl.float32), float(B)), 0).to(
                tl.int32
            )
            best_key = tl.where(most > best_count, start + local, best_key)
            best_count = tl.maximum(best_count, most)
        if V.dtype.element_ty.is_floating():
            bits = tl.where((best_key & 32768) != 0, best_key ^ 32768, ~best_key).to(
                tl.uint16
            )
        else:
            bits = (best_key ^ 32768).to(tl.uint16)
        value = bits.to(V.dtype.element_ty, bitcast=True)
        tl.store(V + row, value)


@libentry()
@triton.jit
def _mode_find_indices_blocked(
    X, V, IPtr, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr
):
    per = tl.cdiv(M, tl.num_programs(0))
    c = tl.arange(0, B)
    for row in range(
        tl.program_id(0) * per, tl.minimum((tl.program_id(0) + 1) * per, M)
    ):
        value = tl.load(V + row)
        if X.dtype.element_ty == tl.float16 or X.dtype.element_ty == tl.bfloat16:
            value = value.to(tl.float32)
        chosen = N
        block = 0
        while (block < tl.cdiv(N, B)) & (chosen == N):
            p = block * B + c
            x = tl.load(X + row * N + p, p < N, other=0)
            if X.dtype.element_ty == tl.float16 or X.dtype.element_ty == tl.bfloat16:
                x = x.to(tl.float32)
            matches = (p < N) & ((x == value) | ((x != x) & (value != value)))
            local = tl.min(tl.where(matches, c.to(tl.float32), float(B)), 0).to(
                tl.int32
            )
            chosen = tl.where(local < B, block * B + local, chosen)
            block += 1
        tl.store(IPtr + row, chosen)


@libentry()
@triton.jit
def _mode_radix_histogram(
    X,
    H,
    M: tl.constexpr,
    N: tl.constexpr,
    NT: tl.constexpr,
    SHIFT: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    total = M * NT
    per = tl.cdiv(total, tl.num_programs(0))
    for tile in range(
        tl.program_id(0) * per, tl.minimum((tl.program_id(0) + 1) * per, total)
    ):
        row = tile // NT
        col = (tile - row * NT) * B + tl.arange(0, B)
        x = tl.load(X + row * N + col, col < N, other=0)
        key = ((convert_to_uint_preverse_order(x, False) >> SHIFT) & (R - 1)).to(
            tl.int32
        )
        bins = tl.arange(0, R)
        hits = (key[None, :] == bins[:, None]) & (col[None, :] < N)
        counts = tl.sum(hits.to(tl.int32), 1)
        tl.store(H + row * R * NT + bins * NT + (tile - row * NT), counts)


@libentry()
@triton.jit
def _mode_radix_prefix(H, P, M: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    per = tl.cdiv(M, tl.num_programs(0))
    c = tl.arange(0, B)
    for row in range(
        tl.program_id(0) * per, tl.minimum((tl.program_id(0) + 1) * per, M)
    ):
        base = 0
        for block in range(tl.cdiv(S, B)):
            pos = block * B + c
            h = tl.load(H + row * S + pos, pos < S, other=0)
            h2 = tl.trans(tl.reshape(h, (B // 32, 32)))
            q = tl.cumsum(h2, 0)
            totals = tl.sum(h2, 0)
            carry = tl.cumsum(totals, 0) - totals
            p = tl.reshape(tl.trans(q + carry[None, :]), (B,)) - h + base
            tl.store(P + row * S + pos, p, pos < S)
            base += tl.sum(h, 0)


@libentry()
@triton.jit
def _mode_radix_scatter(
    X,
    Y,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    NT: tl.constexpr,
    SHIFT: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    total = M * NT
    per = tl.cdiv(total, tl.num_programs(0))
    for tile in range(
        tl.program_id(0) * per, tl.minimum((tl.program_id(0) + 1) * per, total)
    ):
        row = tile // NT
        col = (tile - row * NT) * B + tl.arange(0, B)
        x = tl.load(X + row * N + col, col < N, other=0)
        key = ((convert_to_uint_preverse_order(x, False) >> SHIFT) & (R - 1)).to(
            tl.int32
        )
        bins = tl.arange(0, R)
        hits = (key[None, :] == bins[:, None]) & (col[None, :] < N)
        # Scan the folded, non-innermost axis to avoid a scalar prefix loop.
        folded = tl.trans(tl.reshape(hits.to(tl.int32), (R, B // 32, 32)), (2, 0, 1))
        sums = tl.sum(folded, 0)
        carry = tl.cumsum(sums, 1) - sums
        ranks = (
            tl.reshape(
                tl.trans(tl.cumsum(folded, 0) + carry[None, :, :], (1, 2, 0)), (R, B)
            )
            - 1
        )
        offsets = tl.load(P + row * R * NT + bins * NT + (tile - row * NT))
        pos = tl.sum(tl.where(hits, ranks + offsets[:, None], 0), 0)
        tl.store(Y + row * N + pos, x, col < N)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_ASCEND MODE")
    assert -inp.ndim <= dim < inp.ndim, "Invalid dim"
    if inp.dtype in (torch.int8, torch.uint8) and inp.shape[dim] > 0:
        return _mode_byte(inp, dim, keepdim)
    dim %= inp.ndim
    x = inp.movedim(dim, -1).contiguous()
    n = x.shape[-1]
    rows = x.numel() // n
    values = torch.empty(x.shape[:-1], dtype=inp.dtype, device=inp.device)
    indices = torch.empty(x.shape[:-1], dtype=torch.int64, device=inp.device)
    if rows:
        with torch_device_fn.device(inp.device):
            # Keep bitonic sorting within a small vector tile on Ascend.
            if n <= 64 and inp.dtype != torch.int64:
                _mode_fused[(min(rows, CORE_NUM),)](
                    x, values, rows, n, triton.next_power_of_2(n)
                )
                _mode_find_indices[(min(rows, CORE_NUM),)](
                    x, values, indices, rows, n, triton.next_power_of_2(n)
                )
            elif n >= 256 and inp.dtype in (torch.float16, torch.bfloat16, torch.int16):
                # Bound the histogram workspace to 64 MiB regardless of row count.
                flat_x = x.reshape(rows, n)
                flat_values = values.reshape(-1)
                flat_indices = indices.reshape(-1)
                counts = torch.empty(
                    (min(rows, 256), 65536), device=x.device, dtype=torch.int32
                )
                for start in range(0, rows, 256):
                    batch_rows = min(256, rows - start)
                    batch_x = flat_x[start : start + batch_rows]
                    batch_values = flat_values[start : start + batch_rows]
                    batch_indices = flat_indices[start : start + batch_rows]
                    counts.zero_()
                    _mode_histogram16[(CORE_NUM,)](batch_x, counts, batch_rows, n, 1024)
                    _mode_histogram16_select[(min(batch_rows, CORE_NUM),)](
                        counts, batch_values, batch_rows, 1024
                    )
                    _mode_find_indices_blocked[(min(batch_rows, CORE_NUM),)](
                        batch_x, batch_values, batch_indices, batch_rows, n, 512
                    )
            elif inp.dtype in (torch.int32, torch.float32) and n < (1 << 30):
                # Stable radix passes need only values; recover indices once below.
                block = 512
                bins = 16
                tiles = triton.cdiv(n, block)
                histogram = torch.empty(
                    (rows, bins, tiles), device=x.device, dtype=torch.int32
                )
                prefix = torch.empty_like(histogram)
                source = x
                target = torch.empty_like(x)
                scratch = torch.empty_like(x)
                for shift in range(0, 32, 4):
                    _mode_radix_histogram[(min(rows * tiles, CORE_NUM),)](
                        source, histogram, rows, n, tiles, shift, bins, block
                    )
                    _mode_radix_prefix[(min(rows, CORE_NUM),)](
                        histogram, prefix, rows, bins * tiles, 1024
                    )
                    _mode_radix_scatter[(min(rows * tiles, CORE_NUM),)](
                        source, target, prefix, rows, n, tiles, shift, bins, block
                    )
                    source, target = target, scratch if shift == 0 else source
                _mode_sorted[(min(rows, CORE_NUM),)](
                    source, None, values, None, rows, n, 512
                )
                _mode_find_indices_blocked[(min(rows, CORE_NUM),)](
                    x, values, indices, rows, n, 512
                )
            else:
                sorted_values, sorted_indices = _mode_sort(x, dim=-1)
                _mode_sorted[(min(rows, CORE_NUM),)](
                    sorted_values, sorted_indices, values, indices, rows, n, 512
                )
    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return ModeOut(values, indices)


__all__ = ["mode"]
