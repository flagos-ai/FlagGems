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

from flag_gems.ops.feature_dropout import feature_dropout as default_feature_dropout
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# Deterministic Philox seed shared by the three kernels below.
_RNG_SEED = 1234


@libentry()
@triton.jit
def feature_dropout_uniform_kernel(
    x_ptr,
    out_ptr,
    numel,
    scale,
    p,
    seed,
    SPATIAL: tl.constexpr,
    BLOCK: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    # Requires SPATIAL % BLOCK == 0: every BLOCK-aligned tile lies entirely
    # inside one channel, so a single scalar draw decides the whole tile.
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    ch = start // SPATIAL
    r = tl.rand(seed, ch)
    fac = tl.where(r < p, 0.0, scale)
    if UNMASKED:
        x = tl.load(
            x_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first"
        ).to(tl.float32)
        y = tl.fma(x, fac.to(tl.float32), 0.0)
        tl.store(out_ptr + offs, y, cache_modifier=".cs", eviction_policy="evict_first")
    else:
        m = offs < numel
        x = tl.load(
            x_ptr + offs,
            mask=m,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        y = tl.fma(x, fac.to(tl.float32), 0.0)
        tl.store(
            out_ptr + offs,
            y,
            mask=m,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )


@libentry()
@triton.jit
def feature_dropout_straddle_kernel(
    x_ptr, out_ptr, numel, spatial, scale, p, seed, BLOCK: tl.constexpr
):
    # Tiles can span channel boundaries: the factor is drawn per element but
    # keyed by channel = flat_index // spatial, so all elements of one channel
    # still observe the same Bernoulli decision.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    ch = offs // spatial
    r = tl.rand(seed, ch)
    fac = tl.where(r < p, 0.0, scale)
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = tl.fma(x, fac.to(tl.float32), 0.0)
    tl.store(out_ptr + offs, y, mask=m)


@libentry()
@triton.jit
def feature_dropout_channel1_kernel(
    x_ptr, out_ptr, numel, scale, p, seed, BLOCK: tl.constexpr, UNMASKED: tl.constexpr
):
    # spatial == 1: every element is its own channel, so each element needs an
    # independent draw. One Philox uint32 serves 8 consecutive elements via
    # 4-bit nibbles, cutting RNG cost by 8x while keeping per-element
    # determinism (element e reads nibble (e & 7) of rand(seed, e >> 3)).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    w = tl.randint(seed, offs >> 3)
    n = (w >> ((offs & 7) * 4)) & 0xF
    r = n.to(tl.float32) * 0.0625
    fac = tl.where(r < p, 0.0, scale)
    if UNMASKED:
        x = tl.load(
            x_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first"
        ).to(tl.float32)
        y = tl.fma(x, fac.to(tl.float32), 0.0)
        tl.store(out_ptr + offs, y, cache_modifier=".cs", eviction_policy="evict_first")
    else:
        m = offs < numel
        x = tl.load(
            x_ptr + offs,
            mask=m,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        y = tl.fma(x, fac.to(tl.float32), 0.0)
        tl.store(
            out_ptr + offs,
            y,
            mask=m,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )


def _use_triton_kernel(x: torch.Tensor, p, train) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    if x.ndim < 2:
        return False
    try:
        pv = float(p)
    except Exception:
        return False
    if not 0.0 < pv < 1.0:
        return False
    return True


def _launch(x: torch.Tensor, p: float, out: torch.Tensor):
    numel = x.numel()
    spatial = 1
    for s in x.shape[2:]:
        spatial *= s
    raw_scale = 1.0 / (1.0 - p)
    # MUSA uses a scalar kernel for one value per channel and a vector kernel
    # for larger channels. The vector kernel casts the scalar to the tensor
    # dtype first, while the scalar kernel keeps the fp32 value.
    scale = raw_scale if spatial == 1 else torch.tensor(raw_scale, dtype=x.dtype).item()
    BLOCK = 1024
    unmasked = numel % BLOCK == 0
    grid = (numel // BLOCK,) if unmasked else (triton.cdiv(numel, BLOCK),)
    with torch_device_fn.device(x.device):
        if spatial == 1:
            feature_dropout_channel1_kernel[grid](
                x,
                out,
                numel,
                scale,
                p,
                _RNG_SEED,
                BLOCK=BLOCK,
                UNMASKED=unmasked,
                num_warps=4,
            )
        elif spatial % BLOCK == 0:
            feature_dropout_uniform_kernel[grid](
                x,
                out,
                numel,
                scale,
                p,
                _RNG_SEED,
                SPATIAL=spatial,
                BLOCK=BLOCK,
                UNMASKED=unmasked,
                num_warps=1,
            )
        else:
            feature_dropout_straddle_kernel[grid](
                x,
                out,
                numel,
                spatial,
                scale,
                p,
                _RNG_SEED,
                BLOCK=BLOCK,
                num_warps=4,
            )


def feature_dropout(x: torch.Tensor, p, train=True):
    logger.debug("GEMS_MTHREADS FEATURE_DROPOUT")
    p = float(p)
    if not bool(train) or p == 0.0:
        return x.clone()
    if not _use_triton_kernel(x, p, train):
        return default_feature_dropout(x, p, train)
    if p == 1.0:
        return torch.zeros_like(x)
    out = torch.empty_like(x)
    _launch(x, p, out)
    return out
