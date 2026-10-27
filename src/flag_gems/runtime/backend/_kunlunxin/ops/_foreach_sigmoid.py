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

"""XPU override for ``aten::_foreach_sigmoid`` (out-of-place sigmoid over Tensor[]).

Mirrors the sibling ``_foreach_sigmoid_`` override (in-place) with two fixes:

1. **Fast math.**  ``1 / (1 + tl.exp(-x))`` with the core ``tl.exp`` (LLVM
   Exp2Op native fast path) instead of the generic executor's
   ``tl_extra_shim.exp2`` soft libdevice extern (~15-40x per lane).
2. **Unmasked fast path.**  A homogeneous list of float tensors is processed by
   a single unmasked 2D-grid kernel (axis 0 selects the tensor, axis 1 the
   chunk) so the accesses lower to block-DMA.  The out-of-place variant reads
   the input pointers and writes to freshly allocated outputs.

   Everything else (complex, mixed dtypes/shapes, gappy views, non-divisible
   sizes, integral promotion) falls back to the shared ``foreach_unary``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_complex_math import c_sigmoid
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import (
    check_tensor_list,
    foreach_unary,
    int_to_float,
    is_flat_addressable,
    tl_dtype,
)

from ._foreach_direct import _select_ptr

logger = logging.getLogger(__name__)

# sigmoid rejects complex32 but accepts complex64/complex128, all ints and bool.
_NO_COMPLEX32 = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    + (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)
    + (torch.complex64, torch.complex128)
)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _sigmoid_fast(x):
    # sigmoid(x) = 1 / (1 + e^-x); core tl.exp, not tl_extra_shim.exp2.
    return 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))


@triton.jit
def _foreach_sigmoid_unmasked_kernel(
    in_meta_ptr,
    out_meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over ``NT`` equal-size tensors (out-of-place).

    ``in_meta_ptr``/``out_meta_ptr`` hold the ``NT`` input/output pointers.  The
    grid is ``(NT, n_elements // BLOCK)``; no mask anywhere, which is what lets
    the backend lower the accesses to block-DMA.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    in_ptr = tl.load(in_meta_ptr + t).to(tl.pointer_type(DT))
    out_ptr = tl.load(out_meta_ptr + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + idx, y.to(DT))


@triton.jit
def _foreach_sigmoid_direct_kernel(
    i0, i1, i2, i3, i4, i5, i6, i7,
    i8, i9, i10, i11, i12, i13, i14, i15,
    o0, o1, o2, o3, o4, o5, o6, o7,
    o8, o9, o10, o11, o12, o13, o14, o15,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Direct-pointer variant: scalar pointer args keep block-DMA."""
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    in_ptr = _select_ptr(
        t, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15
    ).to(tl.pointer_type(DT))
    out_ptr = _select_ptr(
        t, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    ).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx).to(tl.float32)
    tl.store(out_ptr + idx, _sigmoid_fast(x).to(DT))


_MAX_DIRECT = 16


def _launch_direct(tensors, dtype, device):
    if len(tensors) > _MAX_DIRECT:
        return None
    n_elements = tensors[0].numel()
    block, num_warps = _pick_block(n_elements, dtype)
    if n_elements % block != 0:
        return None
    outs = [torch.empty_like(t) for t in tensors]
    in_ptrs = [t.data_ptr() for t in tensors]
    in_ptrs += [in_ptrs[0]] * (_MAX_DIRECT - len(in_ptrs))
    out_ptrs = [o.data_ptr() for o in outs]
    out_ptrs += [out_ptrs[0]] * (_MAX_DIRECT - len(out_ptrs))
    grid = (len(tensors), n_elements // block)
    with torch_device_fn.device(device):
        _foreach_sigmoid_direct_kernel[grid](
            *in_ptrs,
            *out_ptrs,
            tl_dtype(dtype),
            block,
            num_warps=num_warps,
            unroll_num=16,
            buffer_size_limit=8192,
            isCloseMemoryAsync=False,
        )
    return outs


def _pick_block(n_elements, dtype):
    """``(block_size, num_warps)``, mirroring the in-place sibling's sweep.

    Every returned block is a power of two, so the caller only takes the fast
    path when ``n_elements % block == 0``.
    """
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
    _foreach_sigmoid_unmasked_kernel[grid](
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
    outs = _launch_direct(tensors, dtype, device)
    if outs is not None:
        return outs
    return _launch_fast(tensors, dtype, device)


def _foreach_sigmoid(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_SIGMOID")
    tensors = check_tensor_list(self)
    outs = _try_fast(tensors)
    if outs is not None:
        return outs
    return foreach_unary(
        tensors,
        _sigmoid_fast,
        complex_fn=c_sigmoid,
        out_dtype_fn=int_to_float,
        allowed_dtypes=_NO_COMPLEX32,
    )
