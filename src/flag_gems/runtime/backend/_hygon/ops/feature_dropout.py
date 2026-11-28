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

"""Feature dropout as a single streaming Triton kernel.

Semantics (matches ``torch.feature_dropout``):

* ``train=False`` or ``p <= 0``: ``out`` is a private copy of ``input``.
* ``p >= 1``: ``out`` is all zeros.
* otherwise every ``(n, c)`` pair of an ``(N, C, ...)`` input gets an
  independent Bernoulli(keep = 1 - p) mask that is constant over the trailing
  spatial extent ``S = numel / (N * C)``; kept values are scaled by
  ``1 / (1 - p)`` (inverted dropout).

The mask is *derived* per program from an integer hash of the global channel
index and a per-call seed, so no mask tensor is materialised and the data path
is exactly one read plus one write per element.

Three details are required for bit-exact agreement with the reference
``channel * scale``; all were established by measurement on the target:

* the hash must use logical (masked) shifts and a mixed multiply - a linear
  hash clusters badly for the 32..64 channel groups used by the tests;
* the scale must be applied as ``where(keep, x * scale, 0)`` with ``x``
  promoted to fp32 (a per-row ``0.0 / scale`` multiplier lets the backend
  contract the multiply into a conversion that changes the result);
* ``torch``'s fp16 scalar multiply rounds the fp32 product to fp16 with ties
  toward zero, while its bf16 path rounds ties to even, so the fp16 output is
  rounded explicitly (bit trick, exact inside the fp16 range) before the store.

The tensor-shape-independent launch plan is cached per (shape, dtype) because
the smallest benchmark shapes are dominated by per-call host work; the RNG seed
is still drawn on every call.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.contiguous import contiguous as _gems_contiguous

logger = logging.getLogger(__name__)
_seed_base = None
_seed_counter = 0
_PLANS = {}


@triton.jit
def _uniform(cids, seed):
    """Uniform fp32 in [0, 1) per channel index (logical-shift integer hash)."""
    h = cids * 1103515245 + seed
    h = h ^ h >> 16 & 65535
    h = h * 1597334677
    h = h ^ h >> 15 & 131071
    return (h >> 8 & 16777215).to(tl.float32) * 5.960464477539063e-08


@triton.jit
def _scaled(x, keep, SCALE, FIX16: tl.constexpr):
    """Inverted-dropout product; FIX16 rounds fp32 -> fp16 ties toward zero."""
    y = tl.where(keep, x * SCALE, 0.0)
    if FIX16:
        b = y.to(tl.int32, bitcast=True) + 4095 & -8192
        y = b.to(tl.float32, bitcast=True)
    return y


@triton.jit(do_not_specialize=["seed", "p"])
def _fd_row_kernel(
    X,
    Y,
    NC,
    S,
    p,
    seed,
    SCALE,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EVEN: tl.constexpr,
    FIX16: tl.constexpr,
):
    """One tile of ``BLOCK_C`` channel rows by ``BLOCK_S`` spatial positions."""
    cids = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    cols = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    keep = _uniform(cids, seed) >= p
    offs = cids[:, None] * S + cols[None, :]
    if EVEN:
        x = tl.load(X + offs, eviction_policy="evict_first").to(tl.float32)
        tl.store(
            Y + offs, _scaled(x, keep[:, None], SCALE, FIX16), cache_modifier=".cs"
        )
    else:
        m = (cids < NC)[:, None] & (cols < S)[None, :]
        x = tl.load(X + offs, mask=m, other=0.0, eviction_policy="evict_first").to(
            tl.float32
        )
        tl.store(
            Y + offs,
            _scaled(x, keep[:, None], SCALE, FIX16),
            mask=m,
            cache_modifier=".cs",
        )


@triton.jit(do_not_specialize=["seed", "p"])
def _fd_flat_kernel(
    X,
    Y,
    numel,
    p,
    seed,
    SCALE,
    BLOCK: tl.constexpr,
    ZERO: tl.constexpr,
    FIX16: tl.constexpr,
):
    """Flat kernel: per-element mask (used when S == 1), copy and zero."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    if ZERO:
        tl.store(Y + offs, tl.zeros([BLOCK], dtype=tl.float32), mask=m)
    else:
        x = tl.load(X + offs, mask=m, other=0.0, eviction_policy="evict_first").to(
            tl.float32
        )
        keep = _uniform(offs, seed) >= p
        tl.store(Y + offs, _scaled(x, keep, SCALE, FIX16), mask=m, cache_modifier=".cs")


def _make_plan(x):
    """Static launch plan for the train path of one (shape, dtype)."""
    numel = x.numel()
    fix16 = x.dtype is torch.float16
    if numel == 0:
        return (2, 0, None, 0, 0, 0, 0, False, fix16, 0, 4)
    nc = x.shape[0] * x.shape[1]
    s = numel // nc
    if s == 1:
        if numel <= 1048576:
            block, warps = (1024, 4)
        else:
            block, warps = (4096, 4)
        return (
            0,
            block,
            ((numel + block - 1) // block,),
            0,
            0,
            0,
            0,
            False,
            fix16,
            numel,
            warps,
        )
    bs = min(triton.next_power_of_2(s), 1024)
    if s >= 1024 and x.dtype is not torch.float16:
        elem, warps = (8192, 8)
    else:
        elem, warps = (4096, 4)
    bc = elem // bs
    if bc > 256:
        bc = 256
    elif bc < 1:
        bc = 1
    even = s % bs == 0 and nc % bc == 0
    grid = ((nc + bc - 1) // bc, (s + bs - 1) // bs)
    return (1, 0, grid, nc, s, bc, bs, even, fix16, numel, warps)


def _next_seed():
    """Fresh 31-bit seed per call (seeded from torch's current generator)."""
    global _seed_base, _seed_counter
    _seed_counter += 1
    if _seed_base is None:
        try:
            _seed_base = int(torch.cuda.initial_seed()) & 4294967295
        except Exception:
            _seed_base = 1575931494
    return _seed_base + _seed_counter * 2654435761 & 2147483647


def _copy(x):
    if not x.is_contiguous():
        x = _gems_contiguous(x)
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        _fd_flat_kernel[triton.cdiv(n, 1024),](
            x, out, n, 0.0, _next_seed(), 1.0, 1024, False, False, num_warps=4
        )
    return out


def _zero(x):
    if not x.is_contiguous():
        x = _gems_contiguous(x)
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        _fd_flat_kernel[triton.cdiv(n, 1024),](
            x, out, n, 1.0, 0, 0.0, 1024, True, False, num_warps=4
        )
    return out


def feature_dropout(input, p, train=True):
    logger.debug("GEMS_HYGON FEATURE_DROPOUT")
    if type(p) is not float:
        if isinstance(p, torch.Tensor):
            p = p.item()
        p = float(p)
    if not train or p <= 0.0:
        return _copy(input)
    if p >= 1.0:
        return _zero(input)
    x = input
    if x.dim() < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    key = (x.shape, x.dtype)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _make_plan(x)
        _PLANS[key] = plan
    if not x.is_contiguous():
        x = _gems_contiguous(x)
    out = torch.empty_like(x)
    kind = plan[0]
    if kind == 2:
        return out
    seed = _next_seed()
    scale = 1.0 / (1.0 - p)
    if kind == 0:
        _fd_flat_kernel[plan[2]](
            x, out, plan[9], p, seed, scale, plan[1], False, plan[8], num_warps=plan[10]
        )
    else:
        _fd_row_kernel[plan[2]](
            x,
            out,
            plan[3],
            plan[4],
            p,
            seed,
            scale,
            plan[5],
            plan[6],
            plan[7],
            plan[8],
            num_warps=plan[10],
        )
    return out
