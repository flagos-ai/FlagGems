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

"""XPU override for ``aten::_foreach_expm1`` / ``aten::_foreach_expm1_``.

A homogeneous list -- one device, one dtype (fp16/bf16/fp32), every tensor
dense/flat-addressable, no internal overlap, all with the same power-of-two
element count -- is processed by one unmasked 2D-grid kernel: without a mask
the loads and stores become wide contiguous accesses.  Everything else goes
through the shared ``foreach_unary``.

An integral or bool input is promoted to its ATen result dtype before the
kernel runs, so the kernel is handed a floating dtype rather than converting
inside the launch; the promoted result matches the CPU reference, which the
converting form does not on this backend.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import UNARY_OPS
from flag_gems.utils.foreach import (
    check_dtype_supported,
    check_tensor_list,
    foreach_unary,
    has_internal_overlap,
    int_to_float,
    is_flat_addressable,
    tl_dtype,
)

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _expm1_fast(x):
    return tl.exp(x.to(tl.float32)) - 1.0


@triton.jit
def _foreach_expm1_unmasked_kernel(
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
    tl.store(ptr + idx, _expm1_fast(x).to(DT))


def _pick_block(n_elements):
    if n_elements >= 65_536:
        return 65_536, 32
    if n_elements >= 16_384:
        return 16_384, 16
    if n_elements >= 4_096:
        return 4_096, 8
    if n_elements >= 1_024:
        return 1_024, 4
    return 128, 4


_META_CACHE_MAX = 8
_meta_cache: dict = {}


def _meta_for(tensors, device):
    """Device-side pointer table for one TensorList.

    The table is a pure function of the tensor addresses, so a repeated call on
    the same list can reuse it instead of re-staging it over PCIe every time.
    """
    key = (device, tuple(t.data_ptr() for t in tensors))
    meta = _meta_cache.get(key)
    if meta is not None:
        return meta
    meta = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64).to(
        device, non_blocking=True
    )
    if len(_meta_cache) >= _META_CACHE_MAX:
        _meta_cache.clear()
    _meta_cache[key] = meta
    return meta


def _launch_fast(tensors, dtype, device):
    n_elements = tensors[0].numel()
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return False
    meta = _meta_for(tensors, device)
    grid = (len(tensors), n_elements // block)
    _foreach_expm1_unmasked_kernel[grid](
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


def _promoted(tensors, allowed):
    """Inputs cast to their ATen result dtype, refusals kept on the input dtype."""
    for t in tensors:
        check_dtype_supported(t.dtype, allowed)
    return [
        t if t.dtype == int_to_float(t.dtype) else t.to(int_to_float(t.dtype))
        for t in tensors
    ]


def _foreach_expm1(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_EXPM1")
    op = UNARY_OPS["expm1"]
    tensors = check_tensor_list(self)
    return foreach_unary(
        _promoted(tensors, op.allowed),
        _expm1_fast,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        allowed_dtypes=op.allowed,
    )


def _foreach_expm1_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_EXPM1_")
    tensors = check_tensor_list(self)
    if _try_fast(tensors):
        return None
    op = UNARY_OPS["expm1"]
    foreach_unary(
        tensors,
        _expm1_fast,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        inplace=True,
        allowed_dtypes=op.allowed,
    )
    return None
