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
from triton.language.extra.xpu.libdevice import log2

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

# Kunlunxin out-of-place `exponential` override.
#
# Why: the generic kernel (src/flag_gems/ops/exponential.py) is compute-bound on
# its per-element `safe_poly_log_f32` (poly5, ~16 FP/int ops per element) plus an
# 8-way unroll with 2 philox calls per program. Measured on P800 (2^28 elements)
# it sustains only ~1.9G elem/s while the vendor engine reaches ~5.8G elem/s and
# the kunlunxin in-place override (exponential_.py, libdevice log2 path, 4-way
# unroll, heuristics-based launch) sustains ~3.2G elem/s. The in-place structure
# is the fastest Triton-side data point on this backend, so the out-of-place
# override mirrors it (BLOCK 256/512/1024 + warps 4/8/16 heuristics, UNROLL=4,
# single philox call per program, log2-based transform).
#
# Two semantic guards kept on purpose:
#   * u is clamped to the smallest fp32 normal before log2: a raw uint32 of 0
#     would otherwise produce log2(0) = -inf -> +inf output (the poly path in
#     the generic kernel clamps too; the in-place kernel has this landmine
#     unguarded).
#   * the `is_min` branch (u >= 1 - eps/2) mirrors the generic/torch epsilon
#     handling so the transform stays positive and avoids log(1-x) precision
#     loss near u == 1.
# For fp64 the transform runs in fp32 and is cast back (libdevice log2 on XPU
# is fp32-only); fp64 is excluded from the functional matrix anyway
# (fp64_is_supported gate) and this keeps the kernel always compilable.


def heur_block(args):
    N = args.get("N", 0)
    if N <= 4096:
        return 256
    elif N <= 65536:
        return 512
    else:
        return 1024


def heur_num_warps(args):
    N = args.get("N", 0)
    if N <= 4096:
        return 4
    elif N <= 65536:
        return 8
    else:
        return 16


@triton.heuristics(
    {
        "BLOCK": heur_block,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def exponential_kernel(
    out_ptr,
    N,
    is_double: tl.constexpr,
    lambd,
    eps,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    if is_double:
        d0 = uint_to_uniform_float(paste_u64(r0, r2))
        d1 = uint_to_uniform_float(paste_u64(r1, r3))
        y0 = transform_exponential(d0, lambd, eps)
        y1 = transform_exponential(d1, lambd, eps)
        UNROLL = 2
        start = tl.program_id(0).to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        tl.store(out_ptr + off_0, y0.to(out_ptr.dtype.element_ty), mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1.to(out_ptr.dtype.element_ty), mask=off_1 < N, eviction_policy="evict_first")
    else:
        f0 = uint_to_uniform_float(r0)
        f1 = uint_to_uniform_float(r1)
        f2 = uint_to_uniform_float(r2)
        f3 = uint_to_uniform_float(r3)
        y0 = transform_exponential(f0, lambd, eps)
        y1 = transform_exponential(f1, lambd, eps)
        y2 = transform_exponential(f2, lambd, eps)
        y3 = transform_exponential(f3, lambd, eps)
        UNROLL = 4
        start = tl.program_id(0).to(tl.uint64) * BLOCK * UNROLL
        off_0 = start + tl.arange(0, BLOCK)
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        tl.store(out_ptr + off_0, y0.to(out_ptr.dtype.element_ty), mask=off_0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, y1.to(out_ptr.dtype.element_ty), mask=off_1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, y2.to(out_ptr.dtype.element_ty), mask=off_2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, y3.to(out_ptr.dtype.element_ty), mask=off_3 < N, eviction_policy="evict_first")


@triton.jit
def paste_u64(hi: tl.uint32, lo: tl.uint32):
    hi = hi.to(tl.uint64) << 32
    x = hi | lo.to(tl.uint64)
    return x


@triton.jit
def transform_exponential(u, lambd, eps):
    # Compute in fp32: XPU libdevice log2 is fp32-only, and this keeps the
    # kernel compilable for the is_double instantiation as well.
    u = u.to(tl.float32)
    eps1 = -0.5 * eps
    is_min = u >= 1.0 + eps1
    trans_scale = 1.0 / 1.4426950408889634
    # NB: the clamp constant must be a tl.full fp32 constant. A bare Python
    # float promotes tl.maximum's result to fp64, and XPU libdevice log2 has
    # no fp64 lowering (it emits a literal "Unsupported" symbol that fails
    # elfconv linking).
    u_safe = tl.maximum(u, tl.full((), 1.17549435e-38, tl.float32))
    log = tl.where(is_min, eps1, log2(u_safe) * trans_scale)
    v = -1.0 / lambd * log
    return v


def exponential(x, lambd: float = 1.0, *, generator=None):
    logger.debug("GEMS_KUNLUNXIN EXPONENTIAL")
    dtype = x.dtype
    device = x.device
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    is_double = dtype in (torch.float64,)
    UNROLL = 2 if is_double else 4
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    eps = torch.finfo(dtype).eps
    res = torch.empty(x.shape, dtype=dtype, device=device)
    if N == 0:
        return res
    with torch_device_fn.device(device):
        exponential_kernel[grid_fn](
            res, N, is_double, lambd, eps, philox_seed, philox_offset
        )
    return res
