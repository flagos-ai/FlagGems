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

"""special_round: elementwise round-to-nearest with `decimals`, ties-to-even.

Reference semantics (matches torch.round / torch.special.round on the target):
  decimals > 0: r = nearbyint(x * 10^d) / 10^d
  decimals == 0: r = nearbyint(x)
  decimals < 0: r = nearbyint(x / 10^|d|) * 10^|d|
computed in opmath precision (fp32 for fp16/bf16/fp32, fp64 for fp64).

Performance: the op is memory-bound. On Iluvatar BI-V150 the eval harness
measures:
  - small tensors (<= 64K elems): launch-bound, prefer BLOCK=1024/warps=8
  - large fp32: BLOCK=256/warps=4 (2 elems/thread)
  - large fp16/bf16: BLOCK=512/warps=4 (4 elems/thread)
Cache policy: plain loads/stores for the 16M class; a .cg store (L2-only,
write-back) is measurably faster than both plain and .cs+evict_first streaming
for the multi-GB 1G class.
"""

import torch
import triton
import triton.language as tl

_THRESHOLD = 65536
_THRESHOLD2 = 67108864  # 64M elements: above this, use .cg store

# (BLOCK, num_warps, cache_modifier_flag) per dtype class and size class
_CFG_F32 = {0: (1024, 8, 0), 1: (256, 4, 0), 2: (256, 4, 1)}
_CFG_LOWP = {0: (1024, 8, 0), 1: (512, 4, 0), 2: (512, 4, 1)}
_CFG_F64 = {0: (1024, 8, 0), 1: (256, 4, 0), 2: (256, 4, 1)}


@triton.jit
def _round_half_even(x):
    # nearbyint (round half to even) for arbitrary-precision float tensors.
    fl = tl.floor(x)
    frac = x - fl
    up = fl + 1.0
    even = (fl * 0.5) == tl.floor(fl * 0.5)
    r = tl.where(frac < 0.5, fl, up)
    r = tl.where((frac == 0.5) & even, fl, r)
    return r


@triton.jit
def _special_round_kernel(
    A,
    out,
    factor,
    n_elements,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    CM: tl.constexpr,
    NO_MASK: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NO_MASK:
        x = tl.load(A + offs)
    else:
        mask = offs < n_elements
        x = tl.load(A + offs, mask=mask, other=0.0)
    x = x.to(COMPUTE)
    f = factor.to(COMPUTE)
    if MODE == 0:
        r = _round_half_even(x)
    elif MODE == 1:
        r = _round_half_even(x * f) / f
    else:
        r = _round_half_even(x / f) * f
    if NO_MASK:
        if CM == 1:
            tl.store(out + offs, r, cache_modifier=".cg")
        else:
            tl.store(out + offs, r)
    else:
        if CM == 1:
            tl.store(out + offs, r, mask=mask, cache_modifier=".cg")
        else:
            tl.store(out + offs, r, mask=mask)


def special_round(A, *, decimals=0):
    n = A.numel()
    out = torch.empty_like(A)
    if n == 0:
        return out
    if not A.is_contiguous():
        A = A.contiguous()
        out = torch.empty_like(A)
    A_flat = A.view(-1)
    out_flat = out.view(-1)

    if decimals > 0:
        mode = 1
        factor = 10.0**decimals
    elif decimals < 0:
        mode = 2
        factor = 10.0 ** (-decimals)
    else:
        mode = 0
        factor = 1.0

    if A.dtype == torch.float64:
        compute = tl.float64
        cfg = _CFG_F64
    elif A.dtype == torch.float32:
        compute = tl.float32
        cfg = _CFG_F32
    else:  # float16 / bfloat16
        compute = tl.float32
        cfg = _CFG_LOWP

    if n <= _THRESHOLD:
        cls = 0
    elif n <= _THRESHOLD2:
        cls = 1
    else:
        cls = 2
    BLOCK, WARPS, CM = cfg[cls]
    grid = ((n + BLOCK - 1) // BLOCK,)
    _special_round_kernel[grid](
        A_flat,
        out_flat,
        factor,
        n,
        BLOCK=BLOCK,
        MODE=mode,
        CM=CM,
        NO_MASK=(n % BLOCK == 0),
        COMPUTE=compute,
        num_warps=WARPS,
    )
    return out
