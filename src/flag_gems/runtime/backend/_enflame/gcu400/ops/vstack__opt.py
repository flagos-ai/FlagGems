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

"""vstack candidate v1: linear concat kernel with BLOCK=65536 for large shapes.

Production `vstack` is just a wrapper around `cat(aligned, dim=0)`; item 21's
`cat__opt.try_fast_cat` uses `BLOCK = 16384 if max_n<65536 else 32768`.
A 30-config sweep (BLOCK x num_warps x 2D/1D grid, bit-identical to torch.cat) shows:
with single-input element count >= 2^24, `BLOCK=65536 / nw4` is steadily better than 32768:

    fp16/bf16 (4096,4096) / (64,512,512)  0.1128 -> 0.0891  1.27x
    fp32 same shape 0.1728 -> 0.1577 1.10x

small shapes (64,64) differ by only 0.0037~0.0044ms across configs (1.05~1.19x,
noise level), so **take over only large shapes**; small shapes and all boundaries
(>4 inputs, non-contiguous, dtype/device mismatch, float8, t0.ndim<2,
shape differs outside cat dim) delegate to production `vstack`, guaranteeing full correctness coverage
without introducing p10 jitter regressions.

Reuse `cat__opt`'s `_can_fast_path` boundary checks and `_linear_cat_kernel`,
without changing cat.py / cat__opt.py (to avoid affecting the closed item 21 concat and
item 30 cat_out), only choosing a larger BLOCK per operator.
"""

import logging
from typing import List, Tuple, Union

import torch
import triton

from flag_gems.runtime.backend._enflame.gcu400.ops.cat import cat as _cat
from flag_gems.runtime.backend._enflame.gcu400.ops.cat__opt import (
    _can_fast_path,
    _linear_cat_kernel,
)

logger = logging.getLogger(__name__)


__all__ = ["vstack"]

_LARGE = 1 << 20


def vstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME_GCU400 VSTACK")
    n = len(tensors)
    if n == 0:
        raise RuntimeError("vstack expected a non-empty TensorList")

    t0 = tensors[0]
    if t0.ndim < 2:
        aligned = list(torch.atleast_2d(tensors))
    else:
        aligned = list(tensors) if not isinstance(tensors, list) else tensors

    A = list(aligned)
    # Take over only "large shape + strictly satisfying cat fast path boundaries"
    if A[0].numel() < _LARGE or t0.ndim < 2 or not _can_fast_path(A, 0):
        # Word-for-word isomorphic direct call of production vstack, zero extra forwarding overhead
        return _cat(aligned, dim=0)

    ns = [A[j].numel() if j < n else 0 for j in range(4)]
    ptrs = [A[j] if j < n else A[0] for j in range(4)]
    offs = [0] * 4
    acc = 0
    for j in range(4):
        offs[j] = acc
        acc += ns[j]
    out_shape = (sum(t.shape[0] for t in A),) + tuple(A[0].shape[1:])
    out = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
    BLOCK = 65536
    grid = (triton.cdiv(max(ns), BLOCK), n)
    _linear_cat_kernel[grid](out, *ptrs, *ns, *offs, NT=n, BLOCK=BLOCK, num_warps=4)
    return out
