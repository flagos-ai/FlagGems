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
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rsub_func(x, y, alpha):
    return y - x * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rsub_func_tensor_scalar(x, y, alpha):
    return y - x * alpha


def rsub_scalar(A, B, alpha=1):
    logger.debug("GEMS_ENFLAME RSUB_SCALAR")
    return rsub_func_tensor_scalar(A, B, alpha)


# ---- flat fast path: same-shape contiguous tensors, one pass over data ----
# Tile policy measured on GCU400 (2026-11-18):
#   * a large BLOCK amortizes per-tile address setup and lifts effective
#     bandwidth from ~1.2 TB/s to ~1.75 TB/s for fp32;
#   * 2-byte dtypes need the full 24-SIP fan-out at large sizes, otherwise the
#     write path stays ~13% below the torch reference;
#   * mid/small tensors prefer a small 8K tile: 16K+ tiles under-fill the
#     pipeline and lose up to 25% in the launch-bound regime.
_LARGE_NUMEL = 8 * 1024 * 1024
_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _select_tile(numel, element_size):
    """Pick (BLOCK, grid cap) from the element count and element width."""
    if numel >= _LARGE_NUMEL:
        return 65536, (24 if element_size == 2 else 12)
    return 8192, 24


@libentry()
@triton.jit(do_not_specialize=["numel"])
def _rsub_flat_kernel(X, Y, Out, numel, alpha, BLOCK: tl.constexpr):
    # Grid-stride loop keeps the grid bounded while every program walks
    # consecutive BLOCK-sized tiles; arithmetic stays in the tensor dtype.
    for tile_index in tl.range(
        tl.program_id(0),
        tl.cdiv(numel, BLOCK),
        tl.num_programs(0),
        num_stages=2,
    ):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        lhs = tl.load(X + offsets, mask=mask, other=0.0)
        rhs = tl.load(Y + offsets, mask=mask, other=0.0)
        tl.store(Out + offsets, rhs - lhs * alpha.to(lhs.dtype), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["num_tiles"])
def _rsub_flat_kernel_exact(X, Y, Out, num_tiles, alpha, BLOCK: tl.constexpr):
    # BLOCK-divisible sizes: dropping the bounds check lets the backend emit
    # unmasked vector loads/stores, worth up to 13% on 2-byte dtypes.
    for tile_index in tl.range(
        tl.program_id(0), num_tiles, tl.num_programs(0), num_stages=2
    ):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        lhs = tl.load(X + offsets)
        rhs = tl.load(Y + offsets)
        tl.store(Out + offsets, rhs - lhs * alpha.to(lhs.dtype))


def rsub_tensor(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME RSUB_TENSOR_FLAT")
    if (
        A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and A.dtype == B.dtype
        and A.dtype in _FAST_DTYPES
        and A.numel() > 0
    ):
        out = torch.empty_like(A)
        numel = A.numel()
        block, sips = _select_tile(numel, A.element_size())
        alpha_f = float(alpha)
        with torch_device_fn.device(A.device):
            if numel % block == 0:
                num_tiles = numel // block
                grid = (min(num_tiles, sips),)
                _rsub_flat_kernel_exact[grid](
                    A.view(-1),
                    B.view(-1),
                    out.view(-1),
                    num_tiles,
                    alpha_f,
                    BLOCK=block,
                    num_warps=4,
                )
            else:
                grid = (min(triton.cdiv(numel, block), sips),)
                _rsub_flat_kernel[grid](
                    A.view(-1),
                    B.view(-1),
                    out.view(-1),
                    numel,
                    alpha_f,
                    BLOCK=block,
                    num_warps=4,
                )
        return out
    return rsub_func(A, B, alpha)
