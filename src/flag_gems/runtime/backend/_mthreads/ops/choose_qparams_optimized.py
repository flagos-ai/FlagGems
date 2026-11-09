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

"""choose_qparams_optimized - fully fused single-launch Triton implementation.

Semantics reproduce the aten CPU reference (verified by a numerical probe):
  * xmin / xmax are the exact fp32 data min / max,
  * stepsize = (xmax - xmin) / n_bins carried in float32,
  * every candidate pair rounds its lower bound and its scale to fp16,
  * q = round_half_to_even((x - xmin_h) / scale) clamped to [0, qmax],
  * loss = sum of squared reconstruction errors in fp32,
  * greedy shrink of whichever side lowers the loss, break when neither does.
The whole min/max pass, the greedy search and the final store run in one CTA
launch, so there is no host sync inside the search loop.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.choose_qparams_optimized import (
    choose_qparams_optimized as default_choose_qparams_optimized,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# The kernel reproduces the aten CPU reference in fp32; other dtypes fall back.
_SUPPORTED_DTYPES = {torch.float32}


@triton.jit
def _half_round(v):
    # Emulate an at::Half cast/round-trip of a float scalar.
    return v.to(tl.float16).to(tl.float32)


@triton.jit
def _round_even(v):
    # round-half-to-even without libdevice (nearbyint semantics).
    f = tl.floor(v)
    d = v - f
    h = f * 0.5
    odd = tl.floor(h) != h
    return tl.where((d > 0.5) | ((d == 0.5) & odd), f + 1.0, f)


@triton.jit
def _loss(x_ptr, n, qmax, lo, hi, BLOCK: tl.constexpr, NCHUNK: tl.constexpr):
    xm = _half_round(lo)
    dr = hi - xm
    sc = tl.where(dr == 0.0, 1.0, _half_round(dr / qmax))
    iv = tl.where(sc == 0.0, 0.0, 1.0 / sc)
    acc = 0.0
    for c in tl.static_range(NCHUNK):
        offs = c * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
        q = _round_even((xv - xm) * iv)
        q = tl.minimum(tl.maximum(q, 0.0), qmax)
        d = xv - (q * sc + xm)
        acc += tl.sum(tl.where(mask, d * d, 0.0))
    return acc


@triton.jit
def _cqo_kernel(
    x_ptr,
    out_ptr,
    n,
    n_bins,
    min_bins,
    qmax,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    mn = float("inf")
    mx = -float("inf")
    for c in tl.static_range(NCHUNK):
        offs = c * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
        mn = tl.minimum(mn, tl.min(tl.where(mask, xv, float("inf"))))
        mx = tl.maximum(mx, tl.max(tl.where(mask, xv, -float("inf"))))

    stepsize = (mx - mn) / n_bins.to(tl.float32)
    threshold = min_bins.to(tl.float32) * stepsize

    cur_min = mn
    cur_max = mx
    cur_loss = _loss(x_ptr, n, qmax, cur_min, cur_max, BLOCK, NCHUNK)

    it = 0
    done = 0
    while (cur_min + threshold < cur_max) & (done == 0) & (it < 1024):
        nmin = cur_min + stepsize
        nmax = cur_max - stepsize
        loss1 = _loss(x_ptr, n, qmax, nmin, cur_max, BLOCK, NCHUNK)
        loss2 = _loss(x_ptr, n, qmax, cur_min, nmax, BLOCK, NCHUNK)
        stop = (cur_loss < loss1) & (cur_loss < loss2)
        take1 = (loss1 < loss2) & (~stop)
        take2 = (loss1 >= loss2) & (~stop)
        cur_min = tl.where(take1, nmin, cur_min)
        cur_max = tl.where(take2, nmax, cur_max)
        cur_loss = tl.where(take1, loss1, tl.where(take2, loss2, cur_loss))
        done = stop.to(tl.int32)
        it += 1

    tl.store(out_ptr, cur_max)
    tl.store(out_ptr + 1, cur_min)


def _use_triton_kernel(input, numel):
    """Whether the call can be handled by the fused MUSA kernel."""
    if not isinstance(input, torch.Tensor):
        return False
    if input.device.type != "musa" or input.dtype not in _SUPPORTED_DTYPES:
        return False
    if int(numel) < 1 or input.numel() < 1:
        return False
    return True


def choose_qparams_optimized(input, numel, n_bins, ratio, bit_width):
    logger.debug("GEMS_MTHREADS CHOOSE_QPARAMS_OPTIMIZED")

    if not _use_triton_kernel(input, numel):
        return default_choose_qparams_optimized(
            input, numel, n_bins, ratio, bit_width
        )

    x = input.view(-1) if input.is_contiguous() else input.reshape(-1)
    n = int(numel)
    nx = x.numel()
    if n > nx:
        n = nx
    nb = int(n_bins)
    min_bins = int(nb * (1.0 - float(ratio)))
    qmax = float((1 << int(bit_width)) - 1)

    block = 1 << (n - 1).bit_length() if n > 1 else 1
    if block > 4096:
        block = 4096
    nchunk = (n + block - 1) // block

    out = torch.empty(2, dtype=torch.float32, device=input.device)
    with torch_device_fn.device(input.device):
        _cqo_kernel[(1,)](
            x,
            out,
            n,
            nb,
            min_bins,
            qmax,
            BLOCK=block,
            NCHUNK=nchunk,
            num_warps=4 if block <= 1024 else 8,
        )
    return out[0:1], out[1:2]
