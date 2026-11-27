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

"""GCU400 floor_: in-place variant on the shared int-domain floor family kernel.

The seeded body lowered ``tl.floor(x.to(tl.float32))`` to an extern libm call
plus an fp32 round-trip on 16-bit inputs behind a fixed B32768/nw2 launch
with masks always on, which kept floor_ at 0.64-1.01x native (arithmean
0.894x, core 15 cases, 2026-11-16) while the sibling out-of-place ops had
already moved to the int-domain family shape at 1.55-1.57x.

This revision reuses the shared ``floor__opt.floor_flat_kernel`` (int-domain
trunc bodies with per-dtype guards, EVEN mask-free arm, int32-safe domain,
cold-compile retry) and only adds an in-place ladder re-scan
(interleaved do_bench median w1000/r100, 3 rounds, 2026-11-16). In-place
sweet points that diverge from the out-of-place ladder are taken only when
they do not collide with the shared LibEntry dispatch key (key = dtype,
alignment, BLOCK, EVEN; num_warps is not part of the key, so two ops sharing
one kernel wrapper must pin a single num_warps per (dtype, BLOCK) pair):

* fp32 > 2^26 wants B65536/nw4 (1334us vs 1340us at 512M, the rsqrt
  in-place retune precedent) on a key the out-of-place ladder never uses
  (it pins fp32 to B32768/nw2);
* bf16 mid (<= 2^24) wants B32768/nw4 (34.6us vs 35.3us at 16.7M), also a
  fresh key (the out-of-place ladder pins bf16 to B65536/nw4);
* fp32 mid stays at B32768/nw2 — the 0.6% gain of nw4 (47.5 vs 47.8us)
  would collide with the out-of-place key, so it is deliberately dropped;
* fp16 arms match the out-of-place ladder exactly (B65536/nw4 mid,
  B131072/nw4 beyond 2^26), as do the tiny (B1024/nw4) and 8K-64K
  (B8192/nw2) bands.

Integral and bool inputs are an identity for floor, so they return without
launching anything (the seeded kernel stored ``x`` back unchanged). Non-
contiguous or oversized (>= int32-safe limit) fp tensors fall back to the
generic pointwise op with an explicit in-place ``out0``; fp64 stays
unsupported on gcu400 (identity, unchanged seed behavior).
"""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .floor__opt import _FLAT_DTYPES, _INT32_SAFE_LIMIT, floor_flat_kernel

logger = logging.getLogger(__name__)

NUM_SIPS = 24


def _choose_launch_inplace(n_total, dtype):
    if n_total <= 8192:
        return 1024, 4
    if n_total <= 65536:
        return 8192, 2
    if dtype == torch.float32:
        if n_total > (1 << 26):
            return 65536, 4
        return 32768, 2
    if dtype == torch.float16:
        if n_total > (1 << 26):
            return 131072, 4
        return 65536, 4
    # bfloat16: fresh mid key up to 2^24, family key beyond
    if n_total <= (1 << 24):
        return 32768, 4
    return 65536, 4


def floor_(input):
    logger.debug("GEMS_ENFLAME FLOOR_")
    x = input
    if not isinstance(x, torch.Tensor):
        raise TypeError("floor_ expects a torch.Tensor.")
    if x.is_complex():
        raise TypeError("floor_ is not supported for complex tensors.")

    if x.dtype not in _FLAT_DTYPES:
        # floor is the identity on integral/bool tensors; fp64 is
        # unsupported on gcu400 (seed behavior: value untouched).
        return x

    n_total = x.numel()
    if n_total == 0:
        return x

    if not x.is_contiguous() or n_total >= _INT32_SAFE_LIMIT:
        from flag_gems.ops.floor import floor_func as _public

        _public(x, out0=x)
        return x

    block, num_warps = _choose_launch_inplace(n_total, x.dtype)
    grid = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    even = n_total % block == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (floor__opt / abs.py /
    # rsqrt.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(x.device):
                floor_flat_kernel[(grid,)](
                    x,
                    x,
                    n_total,
                    BLOCK=block,
                    EVEN=even,
                    IS_FP16=x.dtype == torch.float16,
                    IS_BF16=x.dtype == torch.bfloat16,
                    num_warps=num_warps,
                )
            return x
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "floor_ in-place kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )
    return x
