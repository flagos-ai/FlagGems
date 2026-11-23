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

"""log1p candidate v1: launch config tiered by dtype / element count.

Production `_launch` (11-02 TRACKED) hardcodes `BLOCK=32768 / GRID=48 /
num_warps=4 / NUM_STAGES=4`. A 180-config constexpr sweep (bit-identical to production output)
shows the optimum diverges by dtype:

    fp16/bf16 medium shape (2^24) BS65536 G48 nw2 ST1 1.38x
               large shape (2^29) BS65536 G48 nw2 ST1 1.21x
    fp32 medium shape (2^24) BS8192 G48 nw4 ST4 1.31x
               large shape (2^29) BS8192 G48 nw4 ST1 1.26x
    small shape (4096) shrinking BLOCK to npo2(n) saves microsecond waste (1.5~2.1x)

`NUM_STAGES` matters noticeably for fp16/bf16 large shapes (ST1 1.21x vs ST4 1.11x),
but is nearly irrelevant for fp32 (2.411 vs 2.414), so fp32 keeps production ST4.

Implementation: reuse production `log1p_kernel`'s original JITFunction
(`LibEntry.fn`), passing constexpr explicitly in the per-operator host dispatch;
**TRACKED log1p.py is not modified**, `log1p_` / `log1p_out` are untouched,
and global config is unchanged. Host-side dtype promotion / non-contiguous / empty-tensor semantics
are word-for-word isomorphic to production `log1p()`.

Probe pitfall: `torch.randn()+2.0` still produces x<-1 and NaN from log1p,
and `torch.equal` is always False for NaN, so every config would be misjudged as invalid;
sweeps / gates must use all-positive inputs.
"""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.log1p import (
    log1p_kernel as _kernel_wrapped,
)

logger = logging.getLogger(__name__)


_K = _kernel_wrapped.fn

__all__ = ["log1p"]


def _config(n, dtype):
    """Return (BLOCK_SIZE, GRID_SIZE, num_warps, NUM_STAGES)."""
    if dtype == torch.float32:
        BS, G, nw, ST = 8192, 48, 4, 4
    else:
        BS, G, nw, ST = 65536, 48, 2, 1
    if n < BS:
        BS = max(1024, triton.next_power_of_2(n))
    num_tile = triton.cdiv(n, BS)
    return BS, min(G, num_tile), nw, ST


def log1p(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME_GCU400 LOG1P")
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    n = out.numel()
    if n == 0:
        return out.view_as(x)

    BS, G, nw, ST = _config(n, result_dtype)
    grid = (G, 1, 1)
    with torch_device_fn.device(out.device.index):
        _K[grid](
            x_contig.view(-1),
            out.view(-1),
            n,
            BLOCK_SIZE=BS,
            GRID_SIZE=G,
            NUM_STAGES=ST,
            num_warps=nw,
        )
    return out.view_as(x)
