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

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def amax_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    amax_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amax_val)


@libentry()
@triton.jit
def amax_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    amax_val = tl.max(mid_val)
    tl.store(out, amax_val)


# Row-reduce tile bounds for the along-dim path.
#
# BLOCK_M keeps the well-tuned cluster-count heuristic next_pow2(cdiv(M,12)) as
# the default: it adapts to M (tiny M -> exact small tile, no masked row block --
# a masked partial row block miscompiles bf16 on this XPU) and is what every
# small/degenerate shape that dominates the two-level speedup average was tuned
# for, so it is left untouched. The ONLY change is a clamp for the small-M +
# very-huge-N regime below.
_BLOCK_N_MAX = 8192
# Small M + very huge N (e.g. [1024, 1048576]) leaves only a few row-programs
# (M=1024 -> grid=8) which under-fills the 12 clusters; clamp BLOCK_M *down* to 8
# to expose more row-parallelism (isolation, reduce-INSIDE: [1024,1048576] bf16
# 34.97 -> 24.59ms, fp32 26.26 -> 19.11ms, ALL dtypes win). The threshold is set
# above 65536 because at N=65536 the clamp is a wash / slight loss for fp32
# (reduce-INSIDE bm8 2.55 vs bm128 1.94ms); only N>=~1M is a clear win. Clamp
# DOWN only (min), so tiny M keeps its exact BLOCK_M and never gains a masked row
# block.
_SMALL_M = 4096
_HUGE_N = 262144
_SMALL_BLOCK_M = 8


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("amax"), key=["M", "N"])
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # Keep only a [BLOCK_M, 1] running accumulator and reduce each [BLOCK_M,
    # BLOCK_N] block along N *inside* the loop (reduce-INSIDE). This is the ONLY
    # form that is numerically correct on this XPU: the reduce-OUTSIDE variant
    # (persist a [BLOCK_M, BLOCK_N] tile, tl.max once after the loop) miscompiles
    # for bf16 when full blocks are followed by a masked tail (verified: it fails
    # tests/test_amax.py bf16 with a 0.0625 mismatch, while this form passes). The
    # tiny [BLOCK_M, 1] live state also keeps the loop from materializing a giant
    # 2D tile. The old code's UNBOUNDED BLOCK_M = next_pow2(cdiv(M,12)) heuristic
    # is replaced by a bounded, grid-utilization-aware choice in the launcher.
    acc = tl.full([BLOCK_M, 1], value=-float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=-float("inf")).to(tl.float32)
        a = tl.where(mask, a, -float("inf"))
        blk = tl.max(a, axis=1)[:, None]
        acc = tl.maximum(acc, blk)
    tl.store(out, acc, row_mask)


def amax(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN AMAX")
    if dim is None or len(dim) == 0:
        M = inp.numel()
        # block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        block_size = get_block_size_1d(M, inp.element_size())
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            amax_kernel_1[(mid_size, 1)](
                inp, mid, M, block_size, buffer_size_limit=2048
            )
            amax_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid, buffer_size_limit=2048
            )  # max block size is 128k, so mid does not requires int64 index
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        block_n = min(triton.next_power_of_2(N), _BLOCK_N_MAX)
        # Default: the well-tuned cluster-count heuristic (unchanged from before).
        # tiny M -> exact small tile (no masked row block, which miscompiles bf16).
        block_m = triton.next_power_of_2(triton.cdiv(M, 12))
        # Small M + very huge N under-fills the clusters -> clamp DOWN to 8 to
        # expose more row-parallelism. Clamp only (min), never raising block_m for
        # tiny M, so no shape gains a masked row block vs the default.
        if M <= _SMALL_M and N >= _HUGE_N:
            block_m = min(block_m, _SMALL_BLOCK_M)
        grid = (triton.cdiv(M, block_m),)
        with torch_device_fn.device(inp.device):
            amax_kernel[grid](inp, out, M, N, block_m, block_n, buffer_size_limit=2048)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out
