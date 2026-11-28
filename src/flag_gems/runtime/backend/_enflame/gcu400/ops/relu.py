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

# Flat fast path for contiguous f16/bf16/f32 tensors. The gcu400-local
# pointwise_dynamic codegen (48-CTA cap / tile65536 / nw4 / block-pointer ND
# indexing with boundary_check) measured 706-744us vs native torch relu's
# 679us at 512Mi f16/bf16 and +5-11% at 16Mi on GCU400/L600 (official bench
# core, do_bench kernel median, 2026-11-16). A flat grid-stride kernel with
# grid=min(cdiv, 48), EVEN (n % BLOCK == 0) mask elision and a numel-based
# BLOCK/num_warps ladder reaches native parity in every band (interleaved
# do_bench A/B, same session):
#   big 2^29: 65536/nw2 680.6 (f16, native 681.7) / 678.9 (bf16, native 679.4)
#     / 1375.8 (f32, ~= native); nw4 loses 25-36us.
#   mid 2^24: 65536/nw2 27.5 (f16, was 33.1) / 32768/nw2 48.2 (f32, was 52.5)
#     / 131072/nw4 28.2 (bf16, was 35.4; wider tile hides the bf16 compare
#     cvt chain, same effect as relu6_'s bf16 arm).
#   2^22: 65536/nw2 11.3/12.2 (f16/bf16, was 16.0/17.1).
#   2^20/2^16: 8192-16384/nw2; tiny <=4096: single-CTA next_pow2>=256/nw2
#     (ties the codegen path at 3.1-3.2us).
# Each (dtype, BLOCK) pair maps to exactly one num_warps: LibEntry's dispatch
# key omits num_warps, so a shared BLOCK must never launch with two different
# warp counts. Non-contiguous / non-float / out-of-int32-domain inputs keep
# the pre-change codegen path (gcu400-local pointwise_dynamic fork), which
# restricts flat offsets to < 2^31: the flat kernel is only used when
# numel + BLOCK stays inside the int32-safe domain.
_FLAT_GRID_CAP = 48
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FLAT_MAX_NUMEL = 2**31 - 1 - 65536


@libentry()
@triton.jit(do_not_specialize=["N"])
def _relu_flat_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(N, BLOCK)
    for tile in range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
            tl.store(out_ptr + offs, tl.where(x > 0, x, 0))
        else:
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, tl.where(x > 0, x, 0), mask=mask)


def _relu_flat_config(numel, dtype):
    if numel <= 4096:
        block = 1 << (numel - 1).bit_length() if numel > 1 else 1
        return (256 if block < 256 else block), 2
    if dtype is torch.float32:
        if numel <= 65536:
            return 2048, 2
        if numel <= (1 << 21):
            return 16384, 2
        if numel <= (1 << 26):
            return 32768, 2
    else:
        if numel <= (1 << 21):
            return 8192, 2
        if dtype is torch.bfloat16 and (1 << 22) < numel <= (1 << 26):
            return 131072, 4
    return 65536, 2


def _use_flat_path(A, numel):
    if A.dtype not in _FLAT_DTYPES:
        return False
    if numel == 0 or numel > _FLAT_MAX_NUMEL:
        return False
    return A.is_contiguous()


def _relu_flat_launch(A, out):
    numel = A.numel()
    block, num_warps = _relu_flat_config(numel, A.dtype)
    grid = -(-numel // block)
    if grid > _FLAT_GRID_CAP:
        grid = _FLAT_GRID_CAP
    with torch_device_fn.device(A.device):
        _relu_flat_kernel[(grid,)](
            A,
            out,
            numel,
            BLOCK=block,
            EVEN=(numel % block == 0),
            num_warps=num_warps,
        )


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_forward(x):
    return tl.where(x > 0, x, 0)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_backward(x, dy):
    return tl.where(x > 0, dy, 0.0)


def relu(self):
    logger.debug("GEMS_ENFLAME RELU")
    numel = self.numel()
    if _use_flat_path(self, numel):
        out = torch.empty_like(self)
        _relu_flat_launch(self, out)
        return out
    output = relu_forward(self)
    return output


def relu_(A):
    logger.debug("GEMS_ENFLAME RELU_")
    numel = A.numel()
    if _use_flat_path(A, numel):
        _relu_flat_launch(A, A)
        return A
    out = relu_forward(A, out0=A)
    return out
