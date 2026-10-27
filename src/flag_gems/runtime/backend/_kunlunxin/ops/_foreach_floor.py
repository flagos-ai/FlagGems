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

"""XPU override for ``aten::_foreach_floor`` (out-of-place floor over Tensor[]).

The generic executor's ``UNARY_OPS["floor"]`` lowers ``tl.floor`` to a soft
libdevice extern; the single-tensor ``_kunlunxin/ops/floor.py`` already proved
the magic-number floor is bit-exact for ``|x| < 2^22`` with no extern call.  A
homogeneous list of float tensors additionally gets an unmasked 2D-grid fast
path so the accesses lower to block-DMA.  Everything else falls back to the
shared ``foreach_unary``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import UNARY_OPS
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import (
    check_tensor_list,
    foreach_unary,
    is_flat_addressable,
    tl_dtype,
)

from ._foreach_direct import _select_ptr

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _floor_math(x):
    # magic-number floor: r = RNE(x), d = (r > x), floor = r - d.  No soft
    # extern, no division, no tl.where; bit-exact for |x| < 2^22.
    x_fp32 = x.to(tl.float32)
    r = (x_fp32 + 12582912.0) - 12582912.0
    d = tl.minimum(tl.maximum((r - x_fp32) * 1e38, 0.0), 1.0)
    return r - d


@triton.jit
def _foreach_floor_unmasked_kernel(
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
    x = tl.load(in_ptr + idx)
    y = _floor_math(x)
    tl.store(out_ptr + idx, y.to(DT))


@triton.jit
def _foreach_floor_native_kernel(
    in_meta_ptr,
    out_meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # bf16 fast path: the magic-number floor is ~1.25x slower than the native
    # fp32 ``tl.floor`` on XPU (0.478 vs 0.594 TB/s on [4096,4096]) because the
    # magic math does not vectorize for bf16 loads. bf16 -> fp32 is exact and
    # floor(fp32) of a bf16 value is bit-exact, so the native path is safe. It
    # lives in a separate kernel: fp16/fp32 must NOT reference ``tl.floor``
    # (their extern lowering is a soft libdevice call ~15x slower than magic).
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    in_ptr = tl.load(in_meta_ptr + t).to(tl.pointer_type(DT))
    out_ptr = tl.load(out_meta_ptr + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx).to(tl.float32)
    y = tl.floor(x)
    tl.store(out_ptr + idx, y.to(DT))


@triton.jit
def _foreach_floor_direct_kernel(
    i0, i1, i2, i3, i4, i5, i6, i7,
    i8, i9, i10, i11, i12, i13, i14, i15,
    o0, o1, o2, o3, o4, o5, o6, o7,
    o8, o9, o10, o11, o12, o13, o14, o15,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NATIVE: tl.constexpr,
):
    """Direct-pointer unary launch (in/out selected from 16 scalar slots).

    The meta-pointer variant reloads the address with `tl.load(meta_ptr + t)`,
    which is a runtime global-memory load: the backend cannot apply block-DMA
    to `tl.load(ptr + idx)` even though the address is dense and aligned
    (~840 GB/s vs ~1780 GB/s for a direct pointer). The `tl.where` chain stays
    scalar and analysable.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    in_ptr = _select_ptr(
        t, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15
    ).to(tl.pointer_type(DT))
    out_ptr = _select_ptr(
        t, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    ).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx)
    if NATIVE:
        y = tl.floor(x.to(tl.float32)).to(DT)
    else:
        y = _floor_math(x).to(DT)
    tl.store(out_ptr + idx, y)


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
        _foreach_floor_direct_kernel[grid](
            *in_ptrs,
            *out_ptrs,
            tl_dtype(dtype),
            block,
            dtype == torch.bfloat16,
            num_warps=num_warps,
            unroll_num=16,
            buffer_size_limit=8192,
            isCloseMemoryAsync=False,
        )
    return outs


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
    kernel = (
        _foreach_floor_native_kernel
        if dtype == torch.bfloat16
        else _foreach_floor_unmasked_kernel
    )
    kernel[grid](
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


def _foreach_floor(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_FLOOR")
    tensors = check_tensor_list(self)
    outs = _try_fast(tensors)
    if outs is not None:
        return outs
    op = UNARY_OPS["floor"]
    return foreach_unary(
        tensors,
        op.fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        allowed_dtypes=op.allowed,
    )
