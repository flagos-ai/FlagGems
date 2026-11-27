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

"""softmax_out candidate v1: override constexpr per operator for the K>1 (inner reduction dim) path.

Production gcu400 `softmax_out` (11-03 TRACKED deeply optimized implementation) delegates `K>1`
to common `flag_gems.ops.softmax.softmax_out`, whose
`softmax_kernel_non_inner` uses a global heuristic:

    TILE_K grows greedily (doubles when waves>1, capped at min(K,8192)),
    TILE_N = cdiv(8192, TILE_K)
    num_warps = 4

which pushes TILE_K to 512/1024 on benchmark 3D cases, giving
TILE_N=16/8 and 32/128 serial N-loop iterations per CTA:

    (64,512,512)      fp16 0.247ms  (torch 0.075ms)
    (1024,1024,1024)  fp16 12.16ms (torch 5.35ms)

81 + 14 constexpr sweeps (tolerance-compared with torch.softmax) show
`TILE_K=256 / TILE_N=64 / num_warps=2 / ONE_TILE_PER_CTA=False`
the following is globally optimal and universal:

    (64,512,512)      fp16/bf16 1.46x, fp32 1.27x
    (1024,1024,1024)  fp16 1.37x (production's implicit config measured 1.00x)

Note `num_warps=2` is key to the gain: `nw4` with the same tile is only 0.96x;
`TILE_K<256` or `TILE_N>128` generally degrade, and `TILE_N=256` triggers
OutOfResources(dsm memory).

Implementation: reuse the common kernel's original JITFunction (`LibEntry.fn.fn`)
passing constexpr explicitly, without modifying TRACKED softmax.py or the global
HEURISTICS_CONFIGS.

**Take over only the narrowest safe cases**: floating, contiguous, not half_to_float,
contiguous out with matching dtype, K>1 and K>=256, N>=64; all others
(empty tensors, N==1, K==1, K<256, N<64, integer, half_to_float,
non-contiguous input/out, out-of-range dims other than negative, out aliasing self)
delegate to production `softmax_out`, guaranteeing full correctness coverage.
"""

import logging

import triton

from flag_gems.ops.softmax import softmax_kernel_non_inner as _kernel_wrapped
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.softmax import (
    softmax_out as _prod_softmax_out,
)

logger = logging.getLogger(__name__)


_K = _kernel_wrapped.fn.fn

__all__ = ["softmax_out"]


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_ENFLAME SOFTMAX_OUT")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    if self.numel() == 0:
        return _prod_softmax_out(self, dim, half_to_float, out=out)

    d = dim % self.ndim
    M = 1
    N = self.shape[d]
    for i in range(d):
        M *= self.shape[i]
    Kv = self.numel() // M // N

    if not (
        self.is_floating_point()
        and self.is_contiguous()
        and out.is_contiguous()
        and not half_to_float
        and out.dtype == self.dtype
        and out.data_ptr() != self.data_ptr()
        and Kv > 1
        and Kv >= 256
        and N >= 64
    ):
        return _prod_softmax_out(self, dim, half_to_float, out=out)

    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)

    TILE_K, TILE_N, nw = 256, 64, 2
    grid = (M, triton.cdiv(Kv, TILE_K), 1)
    with torch_device_fn.device(self.device):
        _K[grid](
            out,
            self,
            M,
            N,
            Kv,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            ONE_TILE_PER_CTA=(TILE_N >= N),
            num_warps=nw,
        )
    return out
