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

from flag_gems.ops.gcd_ import gcd_ as default_gcd_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.int16, torch.int32}


@triton.jit
def _gcd_inplace_kernel(
    A,
    B,
    n_elements,
    IS_16: tl.constexpr,
    PEEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.abs(tl.load(A + offsets, mask=mask, other=1)).to(tl.uint32)
    b = tl.abs(tl.load(B + offsets, mask=mask, other=1)).to(tl.uint32)

    if PEEL:
        # Peel one Euclidean step before the loop: this removes the initial
        # block-wide max-reduction from the critical path (the fixed loop-entry
        # latency chain dominates medium/large workloads). The step is
        # idempotent for already-converged lanes (b == 0).
        mx = tl.maximum(a, b)
        mn = tl.minimum(a, b)
        bs = tl.where(mn == 0, 1, mn)
        r = mx % bs
        a = tl.where(mn == 0, mx, mn)
        b = tl.where(mn == 0, 0, r)

    while tl.max(b) > 0:
        mx = tl.maximum(a, b)
        mn = tl.minimum(a, b)
        bs = tl.where(mn == 0, 1, mn)
        r = mx % bs
        a = tl.where(mn == 0, mx, mn)
        b = tl.where(mn == 0, 0, r)

    if IS_16:
        tl.store(A + offsets, a.to(tl.int32).to(tl.int16), mask=mask)
    else:
        tl.store(A + offsets, a.to(tl.int32), mask=mask)


def _specialized_gcd_(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    n_elements = A.numel()
    if n_elements == 0:
        return A
    is_16 = A.dtype.itemsize == 2
    if n_elements <= 16384:
        # Small inputs: keep the per-block reduction shallow and skip the peel
        # (an unconditional extra step costs more than the saved reduction).
        BLOCK_SIZE = 1024
        PEEL = False
    else:
        # Medium/large inputs: peel one step to shorten the loop-entry latency
        # chain. int16 wants 8 elements/thread (BLOCK 2048); int32 wants 8
        # elements/thread at BLOCK 1024 (measured on S5000).
        BLOCK_SIZE = 2048 if is_16 else 1024
        PEEL = True
    num_warps = 8
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _gcd_inplace_kernel[grid](
        A,
        B,
        n_elements,
        IS_16=is_16,
        PEEL=PEEL,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return A


def gcd_(A: torch.Tensor, B: torch.Tensor):
    logger.debug("GEMS_MTHREADS GCD_")
    if (
        isinstance(A, torch.Tensor)
        and A.device.type == "musa"
        and A.dtype in _SUPPORTED_DTYPES
        and isinstance(B, torch.Tensor)
        and B.device.type == "musa"
        and B.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_gcd_(A, B)
    return default_gcd_(A, B)
