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

from flag_gems.ops.sort import sort_stable
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _byte_argsort_small(inp, out, N: tl.constexpr, B: tl.constexpr, DESC: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    v = tl.load(inp + row * N + i, i < N, other=0).to(tl.int32)
    before = v[None, :] > v[:, None] if DESC else v[None, :] < v[:, None]
    before = before | ((v[None, :] == v[:, None]) & (i[None, :] < i[:, None]))
    rank = tl.sum((before & (i[None, :] < N)).to(tl.int32), 1)
    tl.store(out + row * N + rank, i, i < N)


@libentry()
@triton.jit
def _byte_argsort_count(
    inp,
    counts,
    N: tl.constexpr,
    T: tl.constexpr,
    LOW: tl.constexpr,
    DESC: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0) // T
    tile = tl.program_id(0) % T
    bucket = tl.program_id(1)
    value = LOW + (255 - bucket if DESC else bucket)
    i = tile * B + tl.arange(0, B)
    v = tl.load(inp + row * N + i, i < N, other=0).to(tl.int32)
    count = tl.sum(((i < N) & (v == value)).to(tl.int32), 0)
    tl.store(counts + (row * 256 + bucket) * T + tile, count)


@libentry()
@triton.jit
def _byte_argsort_prefix(counts, offsets, T: tl.constexpr, B: tl.constexpr):
    bucket = tl.program_id(0)
    i = tl.arange(0, B)
    count = tl.load(counts + bucket * T + i, i < T, other=0)
    prefix = count if T == 1 else tl.cumsum(count)
    tl.store(offsets + bucket * T + i, prefix - count, i < T)


@libentry()
@triton.jit
def _byte_argsort_bucket_prefix(counts, totals, T: tl.constexpr):
    row = tl.program_id(0)
    bucket = tl.arange(0, 256)
    total = tl.full((256,), 0, tl.int32)
    for tile in range(T):
        total += tl.load(counts + (row * 256 + bucket) * T + tile)
    tl.store(totals + row * 256 + bucket, tl.cumsum(total) - total)


@libentry()
@triton.jit
def _byte_argsort_scatter(
    inp,
    out,
    offsets,
    totals,
    N: tl.constexpr,
    T: tl.constexpr,
    LOW: tl.constexpr,
    DESC: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0) // T
    tile = tl.program_id(0) % T
    bucket = tl.program_id(1)
    value = LOW + (255 - bucket if DESC else bucket)
    i = tile * B + tl.arange(0, B)
    v = tl.load(inp + row * N + i, i < N, other=0).to(tl.int32)
    match = (i < N) & (v == value)
    base = tl.load(totals + row * 256 + bucket)
    base += tl.load(offsets + (row * 256 + bucket) * T + tile)
    rank = tl.cumsum(match.to(tl.int32)) - 1
    tl.store(out + row * N + base + rank, i, match)


def _byte_argsort(inp, dim, descending):
    # Validate the dimension before normalizing negative indices.
    n = inp.shape[dim]
    dim %= inp.ndim
    if n == 1:
        return torch.zeros_like(inp, dtype=torch.int64)
    x = inp.movedim(dim, -1).contiguous()
    out = torch.empty_like(x, dtype=torch.int64)
    if x.numel() == 0:
        return out.movedim(-1, dim)
    rows = x.numel() // n
    with torch_device_fn.device(inp.device):
        if n <= 128:
            _byte_argsort_small[(rows,)](
                x, out, n, triton.next_power_of_2(n), descending
            )
        else:
            # Separate count, prefix and scatter launches avoid inter-block waits.
            block = 512
            tiles = triton.cdiv(n, block)
            counts = torch.empty((rows, 256, tiles), device=x.device, dtype=torch.int32)
            offsets = torch.empty_like(counts)
            totals = torch.empty((rows, 256), device=x.device, dtype=torch.int32)
            low = torch.iinfo(x.dtype).min
            _byte_argsort_count[(rows * tiles, 256)](
                x, counts, n, tiles, low, descending, block
            )
            _byte_argsort_prefix[(rows * 256,)](
                counts, offsets, tiles, triton.next_power_of_2(tiles)
            )
            _byte_argsort_bucket_prefix[(rows,)](counts, totals, tiles)
            _byte_argsort_scatter[(rows * tiles, 256)](
                x, out, offsets, totals, n, tiles, low, descending, block
            )
    return out.movedim(-1, dim)


def argsort(inp, dim=-1, descending=False):
    """Returns the indices that sort a tensor along a given dimension.

    This is equivalent to calling torch.sort and returning only the indices.
    """
    logger.debug("GEMS ARGSORT")
    if inp.dtype in (torch.int8, torch.uint8) and device.vendor_name in (
        "ascend",
        "hygon",
        "mthreads",
    ):
        return _byte_argsort(inp, dim, descending)
    _, indices = sort_stable(inp, stable=True, dim=dim, descending=descending)
    return indices
