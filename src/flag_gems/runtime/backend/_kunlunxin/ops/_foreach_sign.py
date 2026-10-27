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

"""XPU override for ``aten::_foreach_sign`` (out-of-place sign over Tensor[]).

The generic executor's ``_sign_func`` is ``(x > 0).to(DT) - (x < 0).to(DT)``,
which on this backend casts the int1 predicate straight to fp16/bf16 and lowers
to a pathologically slow sequence (~20x, see KERNEL_OPT_EXPERIENCE.md).  The
fast formulation is a plain fp32 ``tl.where`` chain -- no int1 cast, no integer
bit ALU -- plus an unmasked 2D-grid path so the loads/stores lower to block-DMA.
Everything else (complex/int/gappy/non-divisible) falls back to the shared
``foreach_unary``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import UNARY_OPS
from flag_gems.utils.foreach import (
    check_tensor_list,
    foreach_unary,
    is_flat_addressable,
    tl_dtype,
)

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _sign_fast(x):
    # sign(x) in fp32 via two compares + select; NaN -> 0 (both compares false).
    return tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))


@triton.jit
def _foreach_sign_unmasked_kernel(
    in_meta_ptr,
    out_meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    in_ptr = tl.load(in_meta_ptr + t).to(tl.pointer_type(DT))
    out_ptr = tl.load(out_meta_ptr + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx).to(tl.float32)
    y = _sign_fast(x)
    tl.store(out_ptr + idx, y.to(DT))


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
        return None
    in_meta = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64).to(
        device, non_blocking=True
    )
    outs = [torch.empty_like(t) for t in tensors]
    out_meta = torch.tensor([o.data_ptr() for o in outs], dtype=torch.int64).to(
        device, non_blocking=True
    )
    grid = (len(tensors), n_elements // block)
    _foreach_sign_unmasked_kernel[grid](
        in_meta,
        out_meta,
        len(tensors),
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return outs


def _try_fast(tensors):
    if not tensors:
        return None
    first = tensors[0]
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    n_elements = first.numel()
    if n_elements < 128:
        return None
    for t in tensors:
        if t.dtype != dtype or t.device != device:
            return None
        if t.numel() != n_elements:
            return None
        if not is_flat_addressable(t):
            return None
    return _launch_fast(tensors, dtype, device)


def _foreach_sign(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_SIGN")
    tensors = check_tensor_list(self)
    outs = _try_fast(tensors)
    if outs is not None:
        return outs
    op = UNARY_OPS["sign"]
    return foreach_unary(
        tensors,
        op.fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        allowed_dtypes=op.allowed,
    )
