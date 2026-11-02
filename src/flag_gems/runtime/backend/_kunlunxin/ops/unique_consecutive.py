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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry

from .unique import _SCAN_BLOCK, _triton_inclusive_scan

logger = logging.getLogger(__name__)

_UC_BLOCK = _SCAN_BLOCK


@libentry()
@triton.jit
def _ne_consecutive_kernel(
    data_ptr,
    ne_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    offs = pid * BLOCK + r

    live = offs < N
    # Clamp both addresses into [0, N) so no load ever leaves the tensor.
    src = tl.where(live, offs, 0)
    prev_raw = tl.where(offs > 0, offs - 1, 0)
    src_prev = tl.where(prev_raw < N, prev_raw, 0)

    a = tl.load(data_ptr + src)
    b = tl.load(data_ptr + src_prev)

    is_first = tl.where(offs == 0, 1, 0)
    diff = tl.where(a != b, 1, 0)
    has_prev = tl.where(offs > 0, 1, 0)
    ne = is_first + diff * has_prev
    ne = tl.where(live, ne, 0).to(tl.int64)
    tl.store(ne_ptr + offs, ne)


@libentry()
@triton.jit
def _unique_consecutive_finalize_kernel(
    data_ptr,
    cum_ptr,
    ne_ptr,
    out_ptr,
    start_ptr,
    n_unique,
    N,
    BLOCK: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    offs = pid * BLOCK + r
    idx = pid * BLOCK + r

    live = offs < N
    src = tl.where(live, offs, 0)
    a = tl.load(data_ptr + src)
    c = tl.load(cum_ptr + idx)
    ne = tl.load(ne_ptr + idx)

    group = (c - 1).to(tl.int32)
    scratch = n_unique + r
    is_start = ne != 0
    dst = tl.where(is_start, group, scratch)

    # data_out[group] = first element of the group (value: data dtype).
    tl.store(out_ptr + dst, a)

    if return_counts:
        tl.store(start_ptr + dst, offs)


@libentry()
@triton.jit
def _uc_inverse_kernel(
    cum_ptr,
    inv_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < N
    c = tl.load(cum_ptr + offs)
    tl.store(inv_ptr + offs, tl.where(live, c - 1, 0))


@libentry()
@triton.jit
def _uc_run_lengths_kernel(
    start32_ptr,
    counts_ptr,
    N,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    cur = tl.load(start32_ptr + tl.minimum(i, n - 1))
    nxt = tl.load(start32_ptr + tl.minimum(i + 1, n - 1))
    cnt = tl.where(i + 1 < n, nxt - cur, N - cur)
    tl.store(counts_ptr + i, cnt.to(tl.int64), mask=i < n)


def unique_consecutive(
    input: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = None,
):
    logger.debug("GEMS_KUNLUNXIN UNIQUE_CONSECUTIVE")

    if dim is not None:
        raise NotImplementedError(
            "Kunlunxin unique_consecutive currently supports only dim=None"
        )

    # Flatten input for the None dim case
    flat_input = input.ravel()
    num_tasks = flat_input.numel()
    device = flat_input.device

    if num_tasks == 0:
        # Handle empty input
        output = torch.empty(0, dtype=input.dtype, device=device)
        inverse_indices = (
            torch.empty(0, dtype=torch.int64, device=device) if return_inverse else None
        )
        counts = (
            torch.empty(0, dtype=torch.int64, device=device) if return_counts else None
        )
        return output, inverse_indices, counts

    num_ctas = triton.cdiv(num_tasks, _UC_BLOCK)
    num_pad = num_ctas * _UC_BLOCK

    # --- stage 1: group-start flag, int64, padded to whole tiles -------------
    ne = torch.empty(num_pad, dtype=torch.int64, device=device)
    with torch_device_fn.device(device.index):
        _ne_consecutive_kernel[(num_ctas,)](flat_input, ne, num_tasks, BLOCK=_UC_BLOCK)

    # --- stage 2: exclusive/inclusive prefix sum (verified toolkit) ----------
    cum = _triton_inclusive_scan(ne)
    n_unique = int(cum[num_tasks - 1].item())

    data_out = torch.empty(n_unique + _UC_BLOCK, dtype=flat_input.dtype, device=device)
    start_positions = (
        torch.empty(n_unique + _UC_BLOCK, dtype=torch.int32, device=device)
        if return_counts
        else None
    )

    # --- stage 3: scatter outputs + group start offsets ---------------------
    dummy = data_out
    with torch_device_fn.device(device.index):
        _unique_consecutive_finalize_kernel[(num_ctas,)](
            flat_input,
            cum,
            ne,
            data_out,
            start_positions if start_positions is not None else dummy,
            n_unique,
            num_tasks,
            BLOCK=_UC_BLOCK,
            return_counts=return_counts,
        )

    output = data_out[:n_unique]

    # --- stage 4: inverse indices (own kernel -> a single int64 store) ------
    inverse_indices = None
    if return_inverse:
        inverse_indices = torch.empty(num_pad, dtype=torch.int64, device=device)
        with torch_device_fn.device(device.index):
            _uc_inverse_kernel[(num_ctas,)](
                cum, inverse_indices, num_tasks, BLOCK=_UC_BLOCK
            )
        inverse_indices = inverse_indices[:num_tasks].view_as(input)

    # --- stage 5: run lengths (own kernel -> a single int64 store) ----------
    counts = None
    if return_counts:
        counts = torch.empty(n_unique, dtype=torch.int64, device=device)
        with torch_device_fn.device(device.index):
            _uc_run_lengths_kernel[(triton.cdiv(n_unique, _UC_BLOCK),)](
                start_positions[:n_unique],
                counts,
                num_tasks,
                n_unique,
                BLOCK=_UC_BLOCK,
            )

    return output, inverse_indices, counts
