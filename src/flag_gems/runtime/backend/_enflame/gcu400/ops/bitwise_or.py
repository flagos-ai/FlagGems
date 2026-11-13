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
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24
_INT32_MAX = torch.iinfo(torch.int32).max
_FAST_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32)


# GCU400 override of the pointwise_dynamic tensor-tensor path (bitwise_and /
# bitwise_xor precedents).  The generated path is a block-pointer kernel with
# the CTA count locked at min(12, num_tiles) and a 4096-lane tile (code_cache
# source), and bool tensors ride the i1 lowering; on the core bench shapes
# (L600) that costs the in-place __ior__ entry 0.10-0.39x vs torch native.
# The flat grid-stride kernel below saturates the memory floor instead:
# 2-read/1-write flat stream, bool handled through an int8 lane view (0/1
# bytes OR identically), and the eligible pairs only (contiguous, same shape,
# same dtype, int32-safe flat offsets).  Everything exotic (broadcasting,
# strided inputs, mixed tensor dtypes, out-of-int32-range sizes, int64
# tensors) falls back to the generated kernels.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def bitwise_or_flat_kernel(x_ptr, y_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        y = tl.load(y_ptr + off, mask=mask)
        tl.store(out_ptr + off, x | y, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def _choose_block(n_total, itemsize):
    # do_bench(median, warmup=1000ms/rep=100ms) sweep on GCU400 (L600) via the
    # bitwise_xor twin kernel: num_warps=2 beats 4 on every bucket; tiny shapes
    # want the exact pow2 tile, mid shapes peak at 65536 lanes, and only
    # >2^28-element 1/2-byte streams gain from 131072.
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 2)
    if n_total <= 65536:
        return 8192
    if n_total <= 524288:
        return 32768
    if n_total > (1 << 28) and itemsize <= 2:
        return 131072
    return 65536


def _lane_view(t):
    # Bool tensors ride the int8 lane: 0/1 bytes OR identically and the
    # generated i1 lowering is far off the byte-stream memory floor.
    return t.view(torch.int8) if t.dtype == torch.bool else t


def _launch_flat(kern, grid, args, block):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic host-level flake on shared nodes); retrying the
    # launch recompiles and succeeds (rad2deg / atan2_ / sinh precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kern[grid](*args, num_warps=2, BLOCK=block)
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "bitwise_or flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _fast_tensor(t):
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.is_contiguous()
        and t.dtype in _FAST_DTYPES
        and t.numel() <= _INT32_MAX
    )


def _flat_config(n_total, itemsize):
    block = _choose_block(n_total, itemsize)
    if triton.cdiv(n_total, block) * block > _INT32_MAX:
        return None
    return block, (min(triton.cdiv(n_total, block), NUM_SIPS * 2),)


def bitwise_or_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR")
    if (
        _fast_tensor(A)
        and _fast_tensor(B)
        and A.dtype == B.dtype
        and A.shape == B.shape
        and A.device == B.device
    ):
        out = torch.empty_like(A)
        n_total = out.numel()
        if n_total == 0:
            return out
        config = _flat_config(n_total, A.element_size())
        if config is not None:
            block, grid = config
            _launch_flat(
                bitwise_or_flat_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(out), n_total),
                block,
            )
            return out
    return bitwise_or_func(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_")
    if (
        _fast_tensor(A)
        and _fast_tensor(B)
        and A.dtype == B.dtype
        and A.shape == B.shape
        and A.device == B.device
        and (A is B or A.untyped_storage().data_ptr() != B.untyped_storage().data_ptr())
    ):
        n_total = A.numel()
        if n_total == 0:
            return A
        config = _flat_config(n_total, A.element_size())
        if config is not None:
            block, grid = config
            _launch_flat(
                bitwise_or_flat_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(A), n_total),
                block,
            )
            return A
    return bitwise_or_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func_scalar(x, y):
    return x | y


def bitwise_or_scalar(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_SCALAR")
    return bitwise_or_func_scalar(A, B)


def bitwise_or_scalar_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_SCALAR_")
    return bitwise_or_func_scalar(A, B, out0=A)


def bitwise_or_scalar_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_SCALAR_TENSOR")
    return bitwise_or_func_scalar(B, A)
