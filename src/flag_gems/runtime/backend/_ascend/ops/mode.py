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
        index = tl.load(IX + row * N + best_pos)
        tl.store(V + row, value)
        tl.store(out_indices + row, index)


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
