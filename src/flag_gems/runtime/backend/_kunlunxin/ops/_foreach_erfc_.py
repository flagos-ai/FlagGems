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

"""XPU override for ``aten::_foreach_erfc_`` (in-place erfc over Tensor[]).

Mirrors the out-of-place ``_foreach_erfc`` override: fast odd-polynomial math
plus an unmasked 2D-grid fast path for a homogeneous list of float tensors,
with the shared ``foreach_unary`` as the fallback.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import UNARY_OPS
from flag_gems.utils.foreach import (
    check_tensor_list,
    foreach_unary,
    has_internal_overlap,
    is_flat_addressable,
    tl_dtype,
)

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _erfc_math(x):
    x = x.to(tl.float32)
    t = x * x
    p = 2.0958534e-12
    p = p * t + -1.4200718e-10
    p = p * t + 4.475963e-09
    p = p * t + -8.813736e-08
    p = p * t + 1.2336866e-06
    p = p * t + -1.3277817e-05
    p = p * t + 0.00011584865
    p = p * t + -0.00084552215
    p = p * t + 0.005211773
    p = p * t + -0.0268563
    p = p * t + 0.112833545
    p = p * t + -0.37612554
    p = p * t + 1.1283791
    v = x * p
    r = tl.minimum(1.0, tl.maximum(-1.0, v))
    return 1.0 - r


@triton.jit
def _foreach_erfc_unmasked_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + idx)
    y = _erfc_math(x)
    tl.store(ptr + idx, y.to(DT))


def _pick_block(n_elements, dtype):
    if n_elements >= 262_144:
        return 131072, 32
    if n_elements >= 65_536:
        return 32768, 16
    if n_elements >= 16_384:
        return 16384, 16
    if n_elements >= 4_096:
        return 4096, 8
    if n_elements >= 1_024:
        return 1024, 4
    return 128, 4


def _launch_fast(tensors, dtype, device):
    n_elements = tensors[0].numel()
    block, num_warps = _pick_block(n_elements, dtype)
    if n_elements % block != 0:
        return False
    meta = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64).to(
        device, non_blocking=True
    )
    grid = (len(tensors), n_elements // block)
    _foreach_erfc_unmasked_kernel[grid](
        meta,
        len(tensors),
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return True


def _try_fast(tensors):
    if not tensors:
        return False
    first = tensors[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return False
    device = first.device
    n_elements = first.numel()
    if n_elements < 128:
        return False
    for t in tensors:
        if t.dtype != dtype or t.device != device:
            return False
        if t.numel() != n_elements:
            return False
        if not is_flat_addressable(t):
            return False
        if has_internal_overlap(t):
            return False
    return _launch_fast(tensors, dtype, device)


def _foreach_erfc_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ERFC_")
    tensors = check_tensor_list(self)
    if _try_fast(tensors):
        return None
    op = UNARY_OPS["erfc"]
    foreach_unary(
        tensors,
        op.fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        inplace=True,
        allowed_dtypes=op.allowed,
    )
    return None
