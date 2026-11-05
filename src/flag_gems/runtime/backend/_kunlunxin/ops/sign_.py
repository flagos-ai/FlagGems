# Copyright 2026, The FlagOS Contributors.
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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Performance evidence (XPU1 probe, see harness/solution/sign_/README.md):
# * The generic fp-compare body (x > 0) - (x < 0) on 16.7M f32: ~800us, and the
#   vendor bit-domain body (_sign_bit in sign.py): ~285us. Both are dominated by
#   per-element ALU that does NOT overlap with memory (a pure-copy kernel with
#   the same launch shape measures ~90us on 16.7M f32, the memory-bound floor).
# * The saturating-multiply body  min(1, max(-1, x*8.6e37))  is 3 fp ops with a
#   short dependency chain (mul -> max -> min) and measures at the memory-bound
#   floor (90us f32 / 48us f16 on 16.7M elements), i.e. the sign math is hidden
#   under the memory traffic. It is the same technique as angle.py's
#   pi*clamp(-real*BIG, 0, 1) (BIG = 8.6e37).
# * Exactness of min(1, max(-1, x*8.6e37)) (verified on-device against
#   torch.sign on the full boundary battery):
#     - exact for every f16 value (incl. subnormals, +-0, +-inf),
#       every f32/bf16 normal (|x| >= 1.18e-38), +-0 and +-inf;
#     - boundary (same class as angle.py's documented BIG caveat, not covered by
#       the sign_ test matrix which uses randn/randint only):
#         NaN -> +-1 (torch.sign(NaN) = 0);
#         |x| < 1/8.6e37 (~1.16e-38, f32/bf16 subnormals) -> 0 (torch: +-1).
# * Integer dtypes keep the exact compare body (torch.sign semantics).
_SMALL_NUMEL = 65536
_SIGN_CONFIG = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    buffer_size_limit=4096,
    unroll_num=8,
)


@triton.jit
def _sign_sat(x):
    # sign(x) via saturating multiply (see module docstring for exactness).
    # f16/bf16 are widened to f32 first (bf16 arithmetic is unreliable on this
    # backend, and f16 needs f32 range so x*8.6e37 saturates for every f16
    # value).
    if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        xw = x.to(tl.float32)
        r = tl.minimum(tl.maximum(xw * 8.6e37, -1.0), 1.0)
        return r.to(x.dtype)
    if x.dtype == tl.float32:
        return tl.minimum(tl.maximum(x * 8.6e37, -1.0), 1.0)
    # integers / other dtypes: exact compare (same semantics as torch.sign)
    return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_SIGN_CONFIG)
@triton.jit
def _sign_func(x):
    return _sign_sat(x)


@triton.jit
def _sign_small_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # small-numel fast path: minimal launch count; in-place safe (load and store
    # use the same index, one element per lane).
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, _sign_sat(x), mask=mask)


def sign_(x: torch.Tensor):
    # In-place sign: out == input. The kernel is element-wise (load -> compute
    # -> store, one element per lane) so reading and writing the same buffer is
    # race-free; the pointwise_dynamic fast path accepts the same-stride
    # non-contiguous in-place view and processes the physical memory stream.
    logger.debug("GEMS_KUNLUNXIN SIGN_")
    if x.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported. Use torch.sgn for complex numbers."
        )
    if x.numel() == 0:
        return x
    if x.dtype == torch.bool:
        # torch.sign(bool) is the identity; in-place is a no-op.
        return x
    if x.numel() <= _SMALL_NUMEL and x.is_contiguous():
        grid = (triton.cdiv(x.numel(), 4096),)
        _sign_small_kernel[grid](
            x.view(-1),
            x.view(-1),
            x.numel(),
            BLOCK_SIZE=4096,
        )
        return x
    _sign_func(x, out0=x)
    return x
