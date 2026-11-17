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

# Kunlunxin (XPU) override of `flag_gems.ops.put_.put_`.
#
# Same root cause and fix as `_kunlunxin/ops/put.py`: the generic scatter kernel
# enforces the ATen index-bounds contract with PTX inline asm (`@p trap`), which
# the XPU backend cannot compile (`llc ... <inline asm>: error: Cannot deduce
# expr kind`), while `tl.device_assert` is a no-op on XPU.  This override keeps
# the whole index check on device: each program stores one int32 out-of-bounds
# verdict, and the host reads only that tiny verdict buffer back and raises.
#
# The generic validation / dim-collapsing / int32-staging / complex / overlap
# logic is reused by patching the generic module's launcher, so only the kernel
# and launcher live here.
import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# `flag_gems.ops` re-exports `put_` as a *function*, which shadows the submodule
# attribute, so fetch the module object explicitly.
_generic = importlib.import_module("flag_gems.ops.put_")

logger = logging.getLogger(__name__)

_OOB_MSG = "device-side assert: cuda_take_put_kernel() index out of bounds"


@libentry()
@triton.jit(do_not_specialize=["N", "inp_numel"])
def put_check_kernel(index_ptr, verdict_ptr, inp_numel, N, BLOCK_SIZE: tl.constexpr):
    """Device-side bounds verdict for `put_`: one int32 per program.

    Separate from the scatter because XPU3's buffer-size tuner rejects an int64
    `tl.atomic_add` scatter combined with a `tl.max` verdict in one kernel
    ("Failed to tune buffer size").
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    idx = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    wrapped = tl.where(idx < 0, idx + inp_numel, idx)
    index_valid = (wrapped >= 0) & (wrapped < inp_numel)
    oob = tl.max((mask & (~index_valid)).to(tl.int32)) > 0
    tl.store(verdict_ptr + pid, oob.to(tl.int32))


@libentry()
@triton.jit(do_not_specialize=["N", "inp_numel"])
def put_kernel(
    inp_ptr,
    index_ptr,
    source_ptr,
    layout_ptr,
    inp_numel,
    N,
    inp_shape0,
    inp_shape1,
    inp_shape2,
    inp_shape3,
    inp_shape4,
    inp_shape5,
    inp_shape6,
    inp_shape7,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    inp_stride3,
    inp_stride4,
    inp_stride5,
    inp_stride6,
    inp_stride7,
    RANK: tl.constexpr,
    IS_ACCUMULATE: tl.constexpr,
    IS_CONTIGUOUS: tl.constexpr,
    HAS_PRECOMPUTED_OFFSETS: tl.constexpr,
    RANK_DYN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    idx = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    wrapped = tl.where(idx < 0, idx + inp_numel, idx)

    index_valid = (wrapped >= 0) & (wrapped < inp_numel)

    final_mask = mask & index_valid
    idx = tl.where(index_valid, wrapped, 0)

    if HAS_PRECOMPUTED_OFFSETS:
        cur = idx
        inp_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        for i in tl.static_range(RANK_DYN - 1, -1, -1):
            dim_size = tl.load(layout_ptr + i).to(tl.int64)
            dim_stride = tl.load(layout_ptr + RANK_DYN + i).to(tl.int64)
            inp_offsets += (cur % dim_size) * dim_stride
            cur = cur // dim_size
    elif IS_CONTIGUOUS:
        inp_offsets = idx
    else:
        cur = idx
        inp_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        if RANK > 7:
            inp_offsets += (cur % inp_shape7) * inp_stride7
            cur = cur // inp_shape7
        if RANK > 6:
            inp_offsets += (cur % inp_shape6) * inp_stride6
            cur = cur // inp_shape6
        if RANK > 5:
            inp_offsets += (cur % inp_shape5) * inp_stride5
            cur = cur // inp_shape5
        if RANK > 4:
            inp_offsets += (cur % inp_shape4) * inp_stride4
            cur = cur // inp_shape4
        if RANK > 3:
            inp_offsets += (cur % inp_shape3) * inp_stride3
            cur = cur // inp_shape3
        if RANK > 2:
            inp_offsets += (cur % inp_shape2) * inp_stride2
            cur = cur // inp_shape2
        if RANK > 1:
            inp_offsets += (cur % inp_shape1) * inp_stride1
            cur = cur // inp_shape1
        if RANK > 0:
            inp_offsets += cur * inp_stride0

    # All store addresses are in-bounds: out-of-range indices are replaced by
    # 0, so an out-of-bounds index can never write outside the tensor.
    src_val = tl.load(source_ptr + offsets, mask=final_mask, other=0)

    if IS_ACCUMULATE:
        tl.atomic_add(inp_ptr + inp_offsets, src_val, mask=final_mask, sem="relaxed")
    else:
        tl.store(inp_ptr + inp_offsets, src_val, mask=final_mask)


@libentry()
@triton.jit(do_not_specialize=["N", "inp_numel"])
def _wrap_index_kernel(index_ptr, out_ptr, inp_numel, N, BLOCK_SIZE: tl.constexpr):
    """Materialise the non-negative form of ``put_``'s flat index for index_add."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    v = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    v = tl.where(v < 0, v + inp_numel, v)
    tl.store(out_ptr + offsets, v, mask=mask)


def _launch(inp, index_flat, source_flat, accumulate):
    """XPU launcher: generic scatter + on-device bounds verdict read back once."""
    N = index_flat.numel()
    inp_numel = inp.numel()

    if inp.is_contiguous():
        shape, stride, rank = [inp_numel], [1], 1
    else:
        shape, stride = _generic._collapse_dims(list(inp.shape), list(inp.stride()))
        rank = len(shape)

    has_precomputed = rank > _generic.MAX_UNROLL_RANK
    dyn_rank = rank if has_precomputed else 0
    if has_precomputed:
        layout_arg = torch.tensor(shape + stride, dtype=torch.int64, device=inp.device)
    else:
        layout_arg = inp

    shape = (shape + [1] * _generic.MAX_UNROLL_RANK)[:_generic.MAX_UNROLL_RANK]
    stride = (stride + [0] * _generic.MAX_UNROLL_RANK)[:_generic.MAX_UNROLL_RANK]

    BLOCK_SIZE, num_warps = 1024, 4
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    verdict = torch.empty(grid[0], dtype=torch.int32, device=inp.device)

    if accumulate:
        # `tl.atomic_add` loses/duplicates individual updates on XPU3 under
        # contention (measured: one add of magnitude ~0.9 silently dropped out
        # of 96k, and a 512-way duplicate uint8 accumulate was 100% wrong), so
        # the accumulate path is routed through the vendor `index_add`, which
        # has a duplicate-safe, non-atomic segment-reduction path.  A `put_`
        # accumulate on the flattened tensor is exactly `index_add(0, index,
        # source)`.  `inp` may be a strided view (a complex real/imag plane) or
        # the contiguous int32 staging buffer.
        with torch_device_fn.device(inp.device):
            put_check_kernel[grid](
                index_flat,
                verdict,
                inp_numel,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        if any(verdict.tolist()):
            raise RuntimeError(_OOB_MSG)
        import flag_gems

        work = inp if inp.is_contiguous() else inp.contiguous()
        flat = work.reshape(-1)
        wid = torch.empty_like(index_flat)
        with torch_device_fn.device(inp.device):
            _wrap_index_kernel[grid](
                index_flat,
                wid,
                inp_numel,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        added = flag_gems.index_add(flat, 0, wid, source_flat)
        inp.copy_(added.reshape(inp.shape))
        return

    with torch_device_fn.device(inp.device):
        put_check_kernel[grid](
            index_flat,
            verdict,
            inp_numel,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        put_kernel[grid](
            inp,
            index_flat,
            source_flat,
            layout_arg,
            inp_numel,
            N,
            *shape,
            *stride,
            RANK=rank if not has_precomputed else 0,
            IS_ACCUMULATE=accumulate,
            IS_CONTIGUOUS=(rank == 1 and stride[0] == 1),
            HAS_PRECOMPUTED_OFFSETS=has_precomputed,
            RANK_DYN=dyn_rank,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    if any(verdict.tolist()):
        raise RuntimeError(_OOB_MSG)


_generic._launch = _launch

put_ = _generic.put_
