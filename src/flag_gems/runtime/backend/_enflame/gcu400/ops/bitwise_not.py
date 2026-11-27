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

_INT32_MAX = torch.iinfo(torch.int32).max
_FLAT_GRID_CAP = 48
_FAST_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32)

# GCU400 flat fast path for contiguous bool/uint8/int8/int16/int32 tensors.
# The gcu400-local pointwise_dynamic codegen channel (48-CTA cap, 64K tile,
# nw4, block-pointer ND indexing with boundary_check) measured on the official
# core bench shapes (kernel-mode do_bench median, warmup 1000ms / rep 100ms,
# 2026-11-16, GCU400/L600) loses to torch native everywhere except the tiny
# band: int16 16Mi 31.4-33.5us vs native 27.4-28.8 (0.86-0.87x), int32 16Mi
# 50.3-54.8 vs 48.0-50.0 (0.90-0.96x), int16 512Mi 724.7-740.8 vs 676.5-683.8
# (0.92-0.94x).  A flat grid-stride kernel with grid=min(cdiv, 48), EVEN
# (n % BLOCK == 0) mask elision and the numel/itemsize BLOCK/num_warps ladder
# below reaches native parity or better in every band (2-round interleaved
# probe, in-place/out-of-place):
#   4K:    3.11-3.24us (native 4.13-4.34)
#   2^22:  int16 8.92/10.43 (native 11.16/12.15), int32 14.41/15.86 (16.70/17.49)
#   16Mi:  int16 25.34/26.49 (27.22/27.96), int32 46.14/47.38 (47.53/48.49)
#   512Mi: int16 667.85/671.03 (669.72/671.97), int32 1327.9/1329.0 (1333/1339)
# At 16Mi and above BOTH sides sit at the ~3.0TB/s flat-stream DRAM roof
# (4GB int32 512Mi traffic / 1.328ms), so the speedup is pinned at ~1.0x by
# memory physics; the wins concentrate in the sub-16Mi ramp where native is
# still below the roof.  int16 pair-packed int32 word lanes (the XOR-family
# widening trick; bitwise NOT commutes with packing) measured WITHIN-NOISE
# (0.1-0.6% at 16Mi/512Mi) -> denied.  tl.range num_stages 2/4 and num_warps 1
# (in-place family pathology) also WITHIN-NOISE/denied.
# Bool tensors ride an int8 lane view with FLIP=1: torch bool storage keeps
# 0/1 bytes and x ^ 1 swaps them exactly, while ~x would store 0xFE/0xFF into
# a bool buffer.  uint8/int8/int16/int32 use ~x directly.  Non-contiguous
# inputs, int64 tensors (i64 pointer lowering family bug) and numels whose
# padded offsets leave the int32 domain fall back to the pre-change codegen
# path (gcu400-local pointwise_dynamic fork), which stays untouched.
# Each (dtype, BLOCK) pair maps to exactly one num_warps: LibEntry's dispatch
# key omits num_warps, so a shared BLOCK must never launch with two different
# warp counts (relu precedent).


@libentry()
@triton.jit(do_not_specialize=["N"])
def bitwise_not_flat_kernel(
    x_ptr, out_ptr, N, BLOCK: tl.constexpr, FLIP: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(N, BLOCK)
    for tile in tl.range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
            if FLIP:
                tl.store(out_ptr + offs, x ^ 1)
            else:
                tl.store(out_ptr + offs, ~x)
        else:
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask)
            if FLIP:
                tl.store(out_ptr + offs, x ^ 1, mask=mask)
            else:
                tl.store(out_ptr + offs, ~x, mask=mask)


def _flat_config(numel, itemsize):
    if numel <= 4096:
        block = 1 << (numel - 1).bit_length() if numel > 1 else 1
        return (256 if block < 256 else block), 2
    if itemsize == 4:
        if numel <= (1 << 22):
            return 8192, 4
        if numel <= (1 << 24):
            return 32768, 2
        return 65536, 4
    if numel <= (1 << 16):
        return 8192, 2
    if numel <= (1 << 22):
        return 16384, 4
    if numel <= (1 << 26):
        return 65536, 2
    return 131072, 4


def _lane_view(t):
    return t.view(torch.int8) if t.dtype == torch.bool else t


def _use_flat(A, numel):
    block, _ = _flat_config(numel, A.element_size())
    return (
        A.is_cuda
        and A.is_contiguous()
        and A.dtype in _FAST_DTYPES
        and triton.cdiv(numel, block) * block <= _INT32_MAX
    )


def _flat_launch(A, out, numel):
    block, num_warps = _flat_config(numel, A.element_size())
    grid = triton.cdiv(numel, block)
    if grid > _FLAT_GRID_CAP:
        grid = _FLAT_GRID_CAP
    lanes = _lane_view(A)
    out_lanes = lanes if out is A else _lane_view(out)
    flip = A.dtype == torch.bool
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic host-level flake on shared nodes); retrying the
    # launch recompiles and succeeds (bitwise_xor / rad2deg precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                bitwise_not_flat_kernel[(grid,)](
                    lanes,
                    out_lanes,
                    numel,
                    BLOCK=block,
                    FLIP=flip,
                    EVEN=(numel % block == 0),
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "bitwise_not flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    logger.debug("GEMS_ENFLAME BITWISE_NOT")
    numel = A.numel()
    if _use_flat(A, numel):
        out = torch.empty_like(A)
        if numel > 0:
            _flat_launch(A, out, numel)
        return out
    return bitwise_not_func(A)


def bitwise_not_(A):
    logger.debug("GEMS_ENFLAME BITWISE_NOT_")
    numel = A.numel()
    if _use_flat(A, numel):
        if numel > 0:
            _flat_launch(A, A, numel)
        return A
    bitwise_not_func(A, out0=A)
    return A
