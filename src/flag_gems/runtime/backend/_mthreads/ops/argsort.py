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

from flag_gems.ops.argsort import _byte_argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sort import sort_stable

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _argsort_short_rows(
    inp,
    out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(inp + row * N + col, col < N, other=0)
    indices = col
    # Compare original values and indices separately to preserve all int64 bits.
    for stage in tl.static_range(1, LOG_BLOCK + 1):
        for step in tl.static_range(stage - 1, -1, -1):
            partner = col ^ (1 << step)
            other_values = tl.gather(values, partner, 0)
            other_indices = tl.gather(indices, partner, 0)
            valid = indices < N
            other_valid = other_indices < N
            if DESCENDING:
                before = other_values > values
            else:
                before = other_values < values
            equal = other_values == values
            if values.dtype.is_floating():
                is_nan = values != values
                other_nan = other_values != other_values
                if DESCENDING:
                    before = before | (other_nan & ~is_nan)
                else:
                    before = before | (~other_nan & is_nan)
                equal = equal | (is_nan & other_nan)
            before = before | (equal & (other_indices < indices))
            # Padding always follows real elements, including infinities and NaNs.
            before = (other_valid & ~valid) | ((other_valid == valid) & before)
            forward = (col & (1 << stage)) == 0
            lower = (col & (1 << step)) == 0
            swap = tl.where(forward == lower, before, ~before)
            values = tl.where(swap, other_values, values)
            indices = tl.where(swap, other_indices, indices)
    tl.store(out + row * N + col, indices.to(tl.int64), col < N)


@libentry()
@triton.jit
def _canonicalize_float_keys(inp, out, NUMEL: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(inp + offset, offset < NUMEL, other=0)
    dtype: tl.constexpr = values.dtype
    if dtype == tl.float64:
        nan_bits = tl.full((), 0x7FF8000000000000, tl.uint64)
    elif dtype == tl.float32:
        nan_bits = tl.full((), 0x7FC00000, tl.uint32)
    elif dtype == tl.bfloat16:
        nan_bits = tl.full((), 0x7FC0, tl.uint16)
    else:
        nan_bits = tl.full((), 0x7E00, tl.uint16)
    canonical_nan = nan_bits.to(dtype, bitcast=True)
    # Radix keys must treat signed zero and all NaN encodings as equal.
    values = tl.where(values == 0, tl.full((), 0, dtype), values)
    values = tl.where(values != values, canonical_nan, values)
    tl.store(out + offset, values, offset < NUMEL)


def argsort(inp, dim=-1, descending=False):
    logger.debug("GEMS_MTHREADS ARGSORT")
    if inp.ndim == 0:
        if dim not in (-1, 0):
            raise IndexError("Dimension out of range for a scalar tensor")
        return torch.zeros_like(inp, dtype=torch.int64)
    n = inp.shape[dim]
    if inp.numel() == 0 or n == 1:
        return torch.zeros_like(inp, dtype=torch.int64)
    if inp.dtype in (torch.int8, torch.uint8):
        return _byte_argsort(inp, dim, descending)
    if n <= 2048:
        dim %= inp.ndim
        x = inp.movedim(dim, -1).contiguous()
        out = torch.empty_like(x, dtype=torch.int64)
        block = triton.next_power_of_2(n)
        with torch_device_fn.device(inp.device):
            _argsort_short_rows[(x.numel() // n,)](
                x, out, n, block, block.bit_length() - 1, descending
            )
        return out.movedim(-1, dim)
    if inp.dtype.is_floating_point:
        contiguous = inp.contiguous()
        keys = torch.empty_like(contiguous)
        with torch_device_fn.device(inp.device):
            _canonicalize_float_keys[(triton.cdiv(inp.numel(), 1024),)](
                contiguous, keys, inp.numel(), 1024
            )
        inp = keys
    # Use the backend's acquire-based lookback instead of the generic sweep.
    _, indices = sort_stable(inp, stable=True, dim=dim, descending=descending)
    return indices
