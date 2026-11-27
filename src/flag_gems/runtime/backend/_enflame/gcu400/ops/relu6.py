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

# Absolute import (not `..utils.pointwise_dynamic`): the generic
# `flag_gems.utils.pointwise_dynamic` codegen path (`_collect_jit_deps`) drops
# `node.level` when collecting this module's top-level imports, so a relative
# import here would be re-emitted as a bare `from utils...` in the generated
# code_cache file and fail with ModuleNotFoundError on first use of
# `relu6_forward_generic` (same trap as relu6_).
from flag_gems.runtime.backend._enflame.gcu400.utils.pointwise_dynamic import (
    pointwise_dynamic,
)
from flag_gems.utils import libentry
from flag_gems.utils import pointwise_dynamic as _generic_pointwise_dynamic

logger = logging.getLogger(__name__)

# Flat fast path for contiguous f16/bf16/f32 tensors. The pre-change kernel
# (fixed BLOCK=8192/num_warps=4, grid<=48, fp32-roundtrip body via
# `.to(tl.float32)`) measured geomean 0.8795x native on the official bench core
# (do_bench kernel median, 2026-11-16 r0): big 2^29 f16/bf16 1103-1106us vs
# native 684-700 (0.62-0.66x), mid 2^24 f16/bf16 40.0-40.2us vs 33.1-33.2
# (0.82x), f32 near parity. A flat grid-stride kernel with grid=min(cdiv, 48),
# EVEN (n % BLOCK == 0) mask elision, native-dtype min/max body (bit-identical
# to the codegen path incl. NaN/inf; the f32 roundtrip body is the extra tax
# the old kernel paid) and a numel-based BLOCK/num_warps ladder reaches the
# 1R1W copy floor / native parity in every band (interleaved 5-round do_bench
# A/B, same session, 2026-11-16):
#   big 2^29: 65536/nw2 701.1 f16 (native 688.8, floor 681.7) / 689.3 bf16
#     (native 688.0) / 1399.6 f32 (native 1391.5, floor 1383.4); 32768/nw2 and
#     nw4 variants lose or tie (f32 nw4 -0.5% is WITHIN-NOISE and nw4 on the
#     shared BLOCK=65536 would collide with the 16-bit nw2 libentry key).
#   mid 2^24: 65536/nw2 27.1 f16 (floor 27.0, native 33.1) / 32768/nw2 48.2 f32
#     (floor 48.2) / 131072/nw4 27.7 bf16 (floor 27.0; wider tile hides the
#     bf16 cvt compare chain, 65536/nw2 is 28.4).
#   2^22: bf16 65536/nw2 12.3 beats 131072/nw4 13.5, so the bf16 wide arm is
#     restricted to (2^22, 2^26]; f16 65536/nw2 11.3 (native 14.7).
#   tiny <=4096: single-CTA next_pow2>=256/nw2 3.1-3.3us (native 4.2-4.7).
# Each (dtype, BLOCK) pair maps to exactly one num_warps: LibEntry's dispatch
# key omits num_warps, so a shared BLOCK must never launch with two different
# warp counts. Non-contiguous / non-float / out-of-domain inputs fall back to
# the gcu400-local pointwise_dynamic fork, and anything whose numel or max
# flat offset leaves the int32-safe domain (< 2^31) is routed to the generic
# wrapper: the local fork corrupts results at numel == 2^31 (half the
# elements skipped, warm-allocator repro, 2026-11-11 probe on relu6_), while
# the generic wrapper measured correct there and is the pre-change code path.
_FLAT_GRID_CAP = 48
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FLAT_MAX_NUMEL = 2**31 - 1 - 65536
_INT32_LIMIT = 2**31


@libentry()
@triton.jit(do_not_specialize=["N"])
def _relu6_flat_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = tl.cdiv(N, BLOCK)
    for tile in range(pid, num_tiles, num_ctas):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
            tl.store(out_ptr + offs, tl.minimum(tl.maximum(x, 0), 6))
        else:
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, tl.minimum(tl.maximum(x, 0), 6), mask=mask)


def _relu6_flat_config(numel, dtype):
    if numel <= 4096:
        return max(triton.next_power_of_2(numel), 256), 2
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


def _max_flat_offset(A):
    offset = 0
    for size, stride in zip(A.shape, A.stride()):
        if size > 1:
            offset += (size - 1) * stride
    return offset


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu6_forward(x):
    return tl.minimum(tl.maximum(x, 0), 6)


@_generic_pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu6_forward_generic(x):
    return tl.minimum(tl.maximum(x, 0), 6)


def relu6(*args, **kwargs):
    logger.debug("GEMS_ENFLAME RELU6")
    x = (
        args[0]
        if len(args) > 0
        else kwargs.get("input", kwargs.get("self", kwargs.get("x")))
    )
    if x is None:
        raise TypeError(
            "relu6 expects a tensor as the first positional argument or keyword 'input'/'self'/'x'."
        )

    numel = x.numel()
    if _use_flat_path(x, numel):
        out = torch.empty_like(x)
        block, num_warps = _relu6_flat_config(numel, x.dtype)
        grid = (min(triton.cdiv(numel, block), _FLAT_GRID_CAP),)
        with torch_device_fn.device(x.device):
            _relu6_flat_kernel[grid](
                x,
                out,
                numel,
                BLOCK=block,
                EVEN=(numel % block == 0),
                num_warps=num_warps,
            )
        return out
    if numel <= _FLAT_MAX_NUMEL and _max_flat_offset(x) < _INT32_LIMIT:
        return relu6_forward(x)
    return relu6_forward_generic(x)
