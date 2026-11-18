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

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_METAX_SPLIT_LAUNCHERS = {}


@libentry(cache_binding=True)
@triton.jit
def _int_mm_metax_split_partial(a_ptr, b_ptr, partial_ptr, M, K, PART_K: tl.constexpr):
    # The launcher selects contiguous inputs with N=64 and 64-aligned K slices.
    pid = tl.program_id(0)
    part = tl.program_id(1)
    grid_m = tl.cdiv(M, 16)
    rows = (pid % grid_m) * 16 + tl.arange(0, 16)
    cols = (pid // grid_m) * 16 + tl.arange(0, 16)
    safe_rows = tl.minimum(rows, M - 1).to(tl.int64)
    ks = (part * PART_K + tl.arange(0, 64)).to(tl.int64)
    a_ptrs = a_ptr + safe_rows[:, None] * K + ks[None, :]
    b_ptrs = b_ptr + ks[:, None] * 64 + cols[None, :]

    acc = tl.zeros((16, 16), tl.int32)
    for _ in range(0, PART_K // 64):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += 64
        b_ptrs += 64 * 64

    offsets = (part * M + rows[:, None]) * 64 + cols[None, :]
    tl.store(partial_ptr + offsets, acc, mask=rows[:, None] < M)


@libentry(cache_binding=True)
@triton.jit
def _int_mm_metax_split_finish(
    partial_ptr, out_ptr, NUMEL: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.uint32)
    for part in tl.static_range(SPLITS):
        v = tl.load(partial_ptr + part * NUMEL + i, i < NUMEL, 0)
        value += v.to(tl.uint32)
    tl.store(out_ptr + i, value.to(tl.int32), i < NUMEL)


@lru_cache(maxsize=16)
def _is_c550(device):
    return "C550" in torch_device_fn.get_device_name(device)


def _metax_split_device_supported(device):
    # FlagTree lite mode can become active after a normal launcher is cached.
    context = getattr(triton.runtime.jit, "DistributedRtContext", None)
    return _is_c550(device) and (context is None or not context().is_lite_mode)


def _launch_metax_split(self, mat2, out, M, N, K, **compiler_options):
    splits = 8
    partial = torch.empty((splits, M, N), dtype=torch.int32, device=self.device)
    partial_args = (self, mat2, partial, M, K, K // splits)
    finish_args = (partial, out, M * N, splits, 256)
    key = (self.device, M, K)
    # Dtypes, strides and pointer alignment are fixed by the caller.
    launchers = _METAX_SPLIT_LAUNCHERS.get(key)
    if launchers is not None:
        run_partial, run_finish = launchers
        run_partial(*partial_args)
        run_finish(*finish_args)
        return

    partial_grid = (triton.cdiv(M, 16) * 4, splits, 1)
    finish_grid = (triton.cdiv(M * N, 256), 1, 1)
    partial_kernel, _ = _int_mm_metax_split_partial[partial_grid](
        *partial_args, num_warps=4, num_stages=2, **compiler_options
    )
    finish_kernel, _ = _int_mm_metax_split_finish[finish_grid](
        *finish_args, num_warps=4
    )
    if len(_METAX_SPLIT_LAUNCHERS) >= 256:
        _METAX_SPLIT_LAUNCHERS.clear()
    _METAX_SPLIT_LAUNCHERS[key] = (
        partial_kernel[partial_grid],
        finish_kernel[finish_grid],
    )
