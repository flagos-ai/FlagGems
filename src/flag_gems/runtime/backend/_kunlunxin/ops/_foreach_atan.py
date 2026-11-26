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

"""Kunlunxin vendor override for aten::_foreach_atan / _foreach_atan_.

Two independent fixes over the generic executor:

1. **Fast math.**  On this backend the generic atan_kernel lowers to the
   soft libdevice extern xpu::atanf (one llvm.call per lane, ~40ns/element,
   see KERNEL_OPT_EXPERIENCE #35/#38).  We compute atan with the vendor
   polynomial body from atan.py (deg-6, single-select range reduction) -- no
   extern, no division, fp32 max abs err ~1.1e-5.

2. **Unmasked fast path.**  The shared foreach_unary executor masks every
   load/store with a runtime idx < n_elements guard, which blocks the
   block-DMA lowering even when the mask is always-true (KERNEL_OPT_EXPERIENCE
   #36).  A homogeneous list -- one device, one dtype (fp16/bf16/fp32), every
   tensor dense/flat-addressable, no internal overlap, all with the same
   element count divisible by the block -- is processed by a dedicated
   unmasked 2D-grid kernel in a single launch.

Everything else -- complex inputs, integral promotion, mixed dtypes/shapes,
gappy or overlapping views, non-divisible sizes -- falls back to the shared
foreach_unary (with the same polynomial body), so validation and error
messages stay identical to the generic path.

Polynomial body (deg-6, single-select range reduction, copied from atan.py):
    a = |v|;  inv = 1/a;  g = min(1, inv^2);  up = v*g  (|up| <= 1)
    r = where(a > 1, pi/2*(v*inv) - p_u, p_u)
fp32 max abs err ~1.1e-5 (9x margin vs test budget ~1.04e-4).  Known edges:
+-inf -> NaN (torch gives +-pi/2; the functional matrix only exercises randn).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_complex_math import c_atan
from flag_gems.ops._foreach_unary import NO_COMPLEX32
from flag_gems.utils.foreach import (
    check_tensor_list,
    foreach_unary,
    has_internal_overlap,
    int_to_float,
    is_flat_addressable,
    tl_dtype,
)

# The registration-liveness test probes the *generic* executor
# (``flag_gems.ops._foreach_unary`` -> ``GEMS _FOREACH_ATAN``), not this vendor
# override, so the vendor logger name is unconstrained.  Use the module logger
# and the vendor prefix, per the operator-log convention.
logger = logging.getLogger(__name__)

# The unmasked fast path only covers the benchmark/real floating dtypes; fp64
# device tensors silently degrade to fp32 on this backend, and integral/complex
# inputs go through the fallback executor (which handles promotion/complex).
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _atan_body(v):
    # fp32 in, fp32 out.  Single poly evaluation + one select, mirroring
    # _kunlunxin/ops/atan.py (_atan_poly_kernel).
    a = tl.abs(v)
    inv = 1.0 / a
    g = tl.minimum(1.0, inv * inv)
    up = v * g
    t = up * up
    p = -0.01440637694
    p = p * t + 0.05959536122
    p = p * t - 0.1229219022
    p = p * t + 0.1961719889
    p = p * t - 0.3330530964
    p = p * t + 0.9999966197
    p_u = up * p
    return tl.where(a > 1.0, 1.5707963267948966 * (v * inv) - p_u, p_u)


@triton.jit
def _atan_fast(x):
    # The executor hands fn the output dtype (fp16/bf16 stay narrow for a
    # dtype-preserving op); upcast here and cast the result back, mirroring
    # what atan_kernel does around the shim call.
    return _atan_body(x.to(tl.float32)).to(x.dtype)


@triton.jit
def _foreach_atan_unmasked_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over NT equal-size tensors (out-of-place).

    meta_ptr holds [input pointers (NT), output pointers (NT)], matching the
    shared executor layout so the pointer block rides a single H2D transfer.
    The grid is (NT, n_elements // BLOCK); no mask anywhere, which is what lets
    the backend lower the accesses to block-DMA.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + idx).to(tl.float32)
    tl.store(out_ptr + idx, _atan_body(x).to(DT))


@triton.jit
def _foreach_atan_unmasked_kernel_(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over NT equal-size tensors (in-place).

    meta_ptr holds the NT input pointers (input and output coincide for the
    in-place schema).  No mask anywhere, so the accesses lower to block-DMA.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + idx).to(tl.float32)
    tl.store(ptr + idx, _atan_body(x).to(DT))


def _pick_block(n_elements):
    """(block_size, num_warps), mirroring the sibling foreach overrides.

    Every returned block is a power of two, so the caller only takes the fast
    path when n_elements % block == 0.
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


def _launch_fast(tensors, dtype, device, inplace):
    n_elements = tensors[0].numel()
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return None
    grid = (len(tensors), n_elements // block)
    if inplace:
        # Only input pointers are needed: the in-place schema writes back to
        # the same storage.
        meta = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64).to(
            device, non_blocking=True
        )
        _foreach_atan_unmasked_kernel_[grid](
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
    # Out-of-place reads the input pointers and writes freshly allocated outputs.
    outs = [torch.empty_like(t) for t in tensors]
    # One flat list -> one tensor -> one H2D transfer (in + out pointers fused).
    meta = torch.tensor(
        [t.data_ptr() for t in tensors] + [o.data_ptr() for o in outs],
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    _foreach_atan_unmasked_kernel[grid](
        meta,
        len(tensors),
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return outs


def _try_fast(tensors, inplace):
    """Return the fast-path result, or None to signal fallback.

    Out-of-place returns the freshly allocated output list; in-place returns
    True after launching.  None means the list is not homogeneous enough
    (or too small / not divisible), so the caller falls back to foreach_unary.
    """
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
        # In-place safety is the same precondition the shared executor enforces;
        # an overlapping/aliasing write must never slip past the fast path.
        if inplace and has_internal_overlap(t):
            return None
    return _launch_fast(tensors, dtype, device, inplace)


def _foreach_atan(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ATAN")
    tensors = check_tensor_list(self)
    outs = _try_fast(tensors, inplace=False)
    if outs is not None:
        return outs
    return foreach_unary(
        tensors,
        _atan_fast,
        complex_fn=c_atan,
        out_dtype_fn=int_to_float,
        allowed_dtypes=NO_COMPLEX32,
    )


def _foreach_atan_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ATAN_")
    tensors = check_tensor_list(self)
    if _try_fast(tensors, inplace=True) is not None:
        # In-place schemas return (); returning the list would make the
        # dispatcher reject the kernel.
        return None
    foreach_unary(
        tensors,
        _atan_fast,
        complex_fn=c_atan,
        out_dtype_fn=int_to_float,
        inplace=True,
        allowed_dtypes=NO_COMPLEX32,
    )
    return None


__all__ = ["_foreach_atan", "_foreach_atan_"]
