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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rsub_func(x, y, alpha):
    return y - x * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rsub_func_tensor_scalar(x, y, alpha):
    return y - x * alpha


def rsub_tensor(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME RSUB_TENSOR")
    return rsub_func(A, B, alpha)


def rsub_scalar(A, B, alpha=1):
    logger.debug("GEMS_ENFLAME RSUB_SCALAR")
    return rsub_func_tensor_scalar(A, B, alpha)


# ---- B2 flat fast path (2026-11-07) ----
_B2_BLOCK = 1024
_B2_SIPS = 48
# Step 4 B12 tuning (2026-11-08): large shapes benefit from bigger tiles /
# fewer SIPs; keep the original config for small shapes to avoid fp32 regression.
_B2_BLOCK_LARGE = 16384
_B2_SIPS_LARGE = 24
_B2_TILE_THRESHOLD = 65536


@libentry()
@triton.jit(do_not_specialize=["N"])
def _b2_rsub_kernel(X, Y, Out, N, ALPHA, BLOCK: tl.constexpr):
    for block in tl.range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        off = block * BLOCK + tl.arange(0, BLOCK)
        m = off < N
        x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
        y = tl.load(Y + off, mask=m, other=0.0).to(tl.float32)
        r = y - x * ALPHA
        tl.store(Out + off, r.to(Out.dtype.element_ty), mask=m)


_B2_CODEGEN = rsub_tensor


def rsub_tensor(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME RSUB_TENSOR_FLAT")
    if (
        A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and A.dtype == B.dtype
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.numel() > 0
    ):
        out = torch.empty_like(A)
        n = A.numel()
        _large = n > _B2_TILE_THRESHOLD
        _blk = _B2_BLOCK_LARGE if _large else _B2_BLOCK
        _sip = _B2_SIPS_LARGE if _large else _B2_SIPS
        grid = (min(triton.cdiv(n, _blk), _sip),)
        with torch_device_fn.device(A.device):
            _b2_rsub_kernel[grid](
                A.view(-1),
                B.view(-1),
                out.view(-1),
                n,
                alpha,
                BLOCK=_blk,
                num_warps=4,
            )
        return out
    return _B2_CODEGEN(A, B, alpha=alpha)
