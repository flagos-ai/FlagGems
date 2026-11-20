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
_GRID_SIZE = 48  # Fixed program count, tiered after sweeping over M sizes
_NUM_WARPS = 1  # Launch config, fixed after sweeping


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
):
    # Accumulate always in fp32: fp16/bf16 inputs are promoted, fp32 inputs stay fp32
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

    grid_size = min(_GRID_SIZE, M)
    inv_N = 1.0 / N

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
            num_warps=_NUM_WARPS,
        )
    return x, residual
