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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# cos is a pure streaming unary operator: on this machine, large shapes (48 CTA / 4 warps) already reach the
# copy-bandwidth roof
# (pure copy measured at ~203 GB/s), so the kernel uses 1D grid-stride contiguous reads/writes; tuning focuses on
# tile size, CTA cap, and launch overhead for small shapes.
# Parameter sweep results (two sizes: 536870912 / 16777216 elements):
#   fp32     BLOCK=65536, CTA<=48  → 205/202 GB/s (same as the pwd default configuration)
#   fp16/bf16 BLOCK=16384, CTA<=192 → 204/199 GB/s (slightly better than the default)
#   Small shapes use BLOCK=1024 to reduce tail latency (4096 elements: 5.2us → ~4.3us)
_BLOCK_FP32 = 65536
_BLOCK_16BIT = 16384
_MAX_CTAS_FP32 = 48
_MAX_CTAS_16BIT = 192
_SMALL_BLOCK = 1024  # Small shapes: reduce single-CTA tail latency and avoid idle work from large tiles
_SMALL_NUMEL = 1 << 16  # Below this element count, use the small-tile path
_MAX_CTAS_SMALL = 48
_NUM_WARPS = 4
_FAST_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _cos_1d_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(n_elements, BLOCK)
    tile = pid
    while tile < num_tiles:
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # On GCU, tl.cos only accepts fp32/fp64 inputs; narrow-width inputs must be promoted before computation and cast
        # back to the original dtype
        y = tl.cos(x.to(tl.float32)).to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)
        tile += num_ctas


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


def _is_fast_path_applicable(tensor):
    """The fast path requires a contiguous tensor whose dtype GCU supports natively."""
    return tensor.is_contiguous() and tensor.dtype in _FAST_DTYPES


def _launch_cos(inp, out):
    n_elements = inp.numel()
    if n_elements == 0:
        return out
    if n_elements <= _SMALL_NUMEL:
        # launch-bound case: small tiles + fewer CTAs for the shortest tail latency
        block, cta_cap = _SMALL_BLOCK, _MAX_CTAS_SMALL
    elif inp.dtype == torch.float32:
        # For 4-byte elements, a single tile can reach 256KB; 48 CTAs are already enough to saturate bandwidth
        block, cta_cap = _BLOCK_FP32, _MAX_CTAS_FP32
    else:
        # For 2-byte elements, use finer tiles in exchange for more CTA parallelism
        block, cta_cap = _BLOCK_16BIT, _MAX_CTAS_16BIT
    num_tiles = triton.cdiv(n_elements, block)
    grid = (min(num_tiles, cta_cap),)
    _cos_1d_kernel[grid](inp, out, n_elements, block, num_warps=_NUM_WARPS)
    return out


def cos(A):
    logger.debug("GEMS_ENFLAME COS")
    if _is_fast_path_applicable(A):
        return _launch_cos(A, torch.empty_like(A))
    return cos_func(A)


def cos_(A):
    logger.debug("GEMS_ENFLAME COS_")
    if _is_fast_path_applicable(A):
        return _launch_cos(A, A)
    cos_func(A, out0=A)
    return A
