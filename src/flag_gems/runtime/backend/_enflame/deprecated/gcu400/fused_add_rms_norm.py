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
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MAX_BLOCK_N = 16384
_GRID_SIZE = 48  # Fix the program count, with tiers chosen after sweeping over M sizes


def _num_warps(M, N):
    """Select num_warps based on (M, N).

    This operator is in-place (both x and residual are written back), so
    autotune cannot try out configurations and a host-side heuristic is used
    instead. num_warps enters the LibEntry dispatch key as NW_TAG (the dispatch
    key itself does not include num_warps), ensuring different configurations
    do not cross-contaminate each other.
    """
    # Measured conclusions (GCU400, shape ∈ {3,4,5,22,2048} × {4096} ∪ {(2048,2048)}):
    # - For small M (grid=min(48,M) CTAs), a single sub-thread must scan the whole row serially, so instruction issue is
    # the bottleneck,
    #   and nw>=2 can cut small-shape fp32 latency from ~6.6us to ~4.6us;
    # - For large M, throughput already reaches 1.0~1.4 TB/s (close to the bandwidth ceiling), and nw1 is on par with or
    # even better than nw2/4,
    #   so keep nw1 to avoid contention among multiple sub-threads.
    return 1 if M >= 128 else 2


@triton.jit(do_not_specialize=["eps", "inv_N"])
def fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    M,
    N,
    eps,
    inv_N,
    BLOCK_N: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    NW_TAG: tl.constexpr = 1,
):
    # Accumulation always uses fp32: fp16/bf16 inputs are promoted to fp32, and fp32 inputs are already fp32
    cdtype = tl.float32

    pid = ext.program_id(0)
    step = tl.num_programs(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    # In the single-block path weight is identical for all rows, so load it once outside the loop
    w_first = tl.load(w_ptr + cols, mask=mask, other=0.0)

    for row in tl.range(pid, M, step):
        row_in = input_ptr + row * N
        row_res = residual_ptr + row * N

        if SINGLE_BLOCK:
            # One tile covers the whole row: x+r stays in registers, shared by residual and normalization
            x = tl.load(row_in + cols, mask, other=0.0).to(cdtype)
            r = tl.load(row_res + cols, mask, other=0.0).to(cdtype)
            t = x + r
            tl.store(row_res + cols, t, mask=mask)
            var = tl.sum(t * t, axis=0) * inv_N
            rrms = tl.math.rsqrt(var + eps)
            y = (t * rrms * w_first).to(cdtype)
            tl.store(row_in + cols, y, mask=mask)
        else:
            # Row wider than one tile: accumulate variance in chunks, then reread residual to normalize
            x = tl.load(row_in + cols, mask, other=0.0).to(cdtype)
            r = tl.load(row_res + cols, mask, other=0.0).to(cdtype)
            t = x + r
            tl.store(row_res + cols, t, mask=mask)
            var_acc = t * t
            for off in range(BLOCK_N, N, BLOCK_N):
                cols2 = off + tl.arange(0, BLOCK_N)
                mask2 = cols2 < N
                xi = tl.load(row_in + cols2, mask2, other=0.0).to(cdtype)
                ri = tl.load(row_res + cols2, mask2, other=0.0).to(cdtype)
                ti = xi + ri
                tl.store(row_res + cols2, ti, mask=mask2)
                var_acc += ti * ti
            var = tl.sum(var_acc, axis=0) * inv_N
            rrms = tl.math.rsqrt(var + eps)
            y = (t * rrms * w_first).to(cdtype)
            tl.store(row_in + cols, y, mask=mask)
            for off in range(BLOCK_N, N, BLOCK_N):
                cols2 = off + tl.arange(0, BLOCK_N)
                mask2 = cols2 < N
                ti = tl.load(row_res + cols2, mask2, other=0.0).to(cdtype)
                wi = tl.load(w_ptr + cols2, mask=mask2, other=0.0)
                yi = (ti * rrms * wi).to(cdtype)
                tl.store(row_in + cols2, yi, mask=mask2)


@triton.jit(do_not_specialize=["eps", "inv_N"])
def fused_add_rms_norm_kernel_2d(
    input_ptr,
    residual_ptr,
    w_ptr,
    M,
    N,
    eps,
    inv_N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Row-tiled variant: one tile covers BLOCK_M whole rows.

    The row-at-a-time kernel above spends ~8us per row when N is tiny (a
    1-element row is a chain of dependent 4-byte accesses), which dominates
    shapes such as (10000, 1) or (100, 256, 100). Rows are contiguous, so a
    (BLOCK_M, BLOCK_N) tile turns those cases into one streaming load and the
    per-row mean becomes a reduction along axis 1.
    """
    cdtype = tl.float32

    pid = ext.program_id(0)
    step = tl.num_programs(0)

    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    # weight is shared by every row, load it once per program
    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(cdtype)
    rows_off = tl.arange(0, BLOCK_M)
    num_row_tiles = (M + BLOCK_M - 1) // BLOCK_M

    for tile in tl.range(pid, num_row_tiles, step, num_stages=1):
        rows = tile * BLOCK_M + rows_off
        offs = rows[:, None] * N + cols[None, :]
        mask = (rows < M)[:, None] & col_mask[None, :]
        x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(cdtype)
        r = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(cdtype)
        t = x + r
        tl.store(residual_ptr + offs, t, mask=mask)
        var = tl.sum(t * t, axis=1) * inv_N
        rrms = tl.math.rsqrt(var + eps)
        y = t * rrms[:, None] * w[None, :]
        tl.store(input_ptr + offs, y.to(x.dtype), mask=mask)


# (BLOCK_M, minimum M) per BLOCK_N, measured on GCU400:
# the wide 512-column tile only pays off once M is large enough to fill the
# machine, below that the row-at-a-time kernel is up to 2.6x faster.
_TILED_ROW_TABLE = {
    1: (512, 1024),
    2: (512, 1024),
    4: (256, 1024),
    8: (256, 1024),
    16: (256, 1024),
    32: (256, 1024),
    64: (128, 1024),
    128: (128, 1024),
    256: (128, 1024),
    512: (64, 4096),
    1024: (32, 512),
    2048: (16, 4096),
    4096: (8, 4096),
}


def _pick_tiled_rows(M, N):
    """Return (BLOCK_M, BLOCK_N) for the row-tiled kernel, or None."""
    block_n = triton.next_power_of_2(N)
    entry = _TILED_ROW_TABLE.get(block_n)
    if entry is None:
        return None
    block_m, min_rows = entry
    if M < min_rows:
        return None
    # keep at least a handful of waves so the 48 fixed CTAs stay busy
    if triton.cdiv(M, block_m) < 8:
        return None
    return block_m, block_n


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    logger.debug(
        "GEMS_ENFLAME FUSED_ADD_RMS_NORM, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_N = min(triton.next_power_of_2(N), MAX_BLOCK_N)
    single_block = triton.next_power_of_2(N) <= MAX_BLOCK_N

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    inv_N = 1.0 / N
    tiled = _pick_tiled_rows(M, N)
    if tiled is not None:
        block_m, tiled_n = tiled
        tiled_grid = min(_GRID_SIZE, triton.cdiv(M, block_m))
        with torch_device_fn.device(x.device):
            fused_add_rms_norm_kernel_2d[(tiled_grid,)](
                x,
                residual,
                weight,
                M,
                N,
                eps,
                inv_N,
                BLOCK_M=block_m,
                BLOCK_N=tiled_n,
                num_warps=1,
            )
        return x, residual

    grid_size = min(_GRID_SIZE, M)
    nw = _num_warps(M, N)

    with torch_device_fn.device(x.device):
        fused_add_rms_norm_kernel[(grid_size,)](
            x,
            residual,
            weight,
            M,
            N,
            eps,
            inv_N,
            BLOCK_N=BLOCK_N,
            SINGLE_BLOCK=single_block,
            NW_TAG=nw,
            num_warps=nw,
        )
    return x, residual
