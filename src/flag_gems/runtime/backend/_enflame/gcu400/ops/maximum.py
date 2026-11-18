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

from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_PERSIST_MIN_N = 2**16
_PERSIST_MAX_N_F32 = 2**28
_BLOCK = 65536
_GRID_CAP = 48
_INT32_SAFE_LIMIT = 0x7FFFFFFF


@triton.jit
def _maximum_persist_kernel(X, Y, OUT, N, BLOCK: tl.constexpr, UPCAST: tl.constexpr):
    # Grid-stride persistent flat kernel (L600, 48 SIPs): tile_size 65536 @
    # num_warps=2 reaches ~3.0 TB/s on 2-byte streams vs ~2.1-2.6 TB/s for the
    # generated bptr/GSL wrapper (bench core, kernel-mode do_bench median).
    pid = tl.program_id(0)
    num = tl.num_programs(0)
    step = num * BLOCK
    for base in range(pid * BLOCK, N, step):
        off = base + tl.arange(0, BLOCK)
        m = off < N
        x = tl.load(X + off, mask=m, other=0)
        y = tl.load(Y + off, mask=m, other=0)
        if UPCAST:
            x = x.to(tl.float32)
            y = y.to(tl.float32)
        r = tl.maximum(x, y)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def maximum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)

    return tl.maximum(X, Y)


def maximum(X, Y):
    logger.debug("GEMS_ENFLAME MAXIMUM")
    assert X.device.type == device and Y.device.type == device
    # Fast path: same-shape contiguous float inputs. fp32 n >= 2**28 stays on
    # the generated wrapper (measured even-or-better there, interleaved A/B),
    # everything below the threshold dispatches to the persistent flat kernel.
    if (
        X.dtype == Y.dtype
        and X.dtype in _FLOAT_DTYPES
        and X.shape == Y.shape
        and X.is_contiguous()
        and Y.is_contiguous()
    ):
        n = X.numel()
        if (
            n >= _PERSIST_MIN_N
            and n <= _INT32_SAFE_LIMIT - _BLOCK
            and not (X.dtype == torch.float32 and n >= _PERSIST_MAX_N_F32)
        ):
            out = torch.empty_like(X)
            grid = (min(_GRID_CAP, triton.cdiv(n, _BLOCK)),)
            _maximum_persist_kernel[grid](
                X,
                Y,
                out,
                n,
                BLOCK=_BLOCK,
                UPCAST=X.dtype == torch.bfloat16,
                num_warps=2,
            )
            return out
    return maximum_kernel(X, Y)
