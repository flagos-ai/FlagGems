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
"""Kunlunxin XPU3 override for ``aten::_convert_indices_from_coo_to_csr``.

The generic implementation (``src/flag_gems/ops/convert_indices_from_coo_to_csr.py``)
is correct on NVIDIA but its ``@triton.jit`` kernel
``_coo_to_csr_lower_bound_kernel`` mis-lowers on XPU3 in two independent ways:

1. **Non-strict comparison against a runtime scalar in the tail mask is
   dropped.** ``tl.store(..., mask=k <= size)`` degenerates into an unmasked
   full-tile store: a ``BLOCK_K=128`` program writes all 128 lanes into a
   ``size + 1`` window (heap OOB write past the buffer).
2. **An all-false masked ``tl.load`` aborts the program before later stores.**
   The host passes ``steps = max(n.bit_length(), 1)``, so with an empty index
   vector (``n == 0``) the bisection loop still runs once and ``mid < n`` is
   false on every lane. The load terminates the program, so the pointer table
   is left completely unwritten (reads ``torch.empty`` garbage).

Fix (validated on XPU3): materialise ``nrow = size + 1`` and use a strict
``k < nrow`` mask, clamp inactive store offsets into the buffer, and guard the
bisection behind ``if n > 0:`` so no all-false load is ever emitted.

The host wrapper in the generic module resolves this kernel as a module global
at launch time, so rebinding the attribute here is sufficient and the generic
source stays untouched (same precedent as ``take_along_dim.py``).
"""

import sys

import triton
import triton.language as tl


@triton.jit
def _coo_to_csr_lower_bound_kernel(
    idx_ptr,
    out_ptr,
    n,
    size,
    stride,
    steps,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    k = pid * BLOCK_K + tl.arange(0, BLOCK_K)

    lo = tl.zeros([BLOCK_K], dtype=tl.int32)
    # Guard the bisection: with n == 0 every lane's ``mid < n`` mask would be
    # all-false and the masked load would abort the program before the store.
    # The lower bound of every key in an empty vector is 0, so ``lo`` already
    # holds the correct pointer table.
    if n > 0:
        hi = tl.zeros([BLOCK_K], dtype=tl.int32) + n
        for _ in range(steps):
            mid = (lo + hi) >> 1
            m = mid < n
            v = tl.load(idx_ptr + mid * stride, mask=m, other=0)
            c = m & (v.to(tl.int32) < k)
            lo = tl.where(c, mid + 1, lo)
            hi = tl.where(c, hi, mid)

    # Strict ``<`` on a materialised bound: XPU3 drops a non-strict
    # ``k <= runtime_scalar`` mask and turns the store into a full-tile write.
    nrow = size + 1
    in_bounds = k < nrow
    # Clamp inactive store offsets into the buffer (offset 0 always exists
    # because size >= 0 => nrow >= 1) so no OOB address enters the vector
    # store; the mask still decides which lanes are actually written.
    store_off = tl.where(in_bounds, k, 0)
    tl.store(out_ptr + store_off, lo.to(out_ptr.dtype.element_ty), mask=in_bounds)


def _patch_generic_kernel():
    """Route the generic host's kernel lookup to this backend-local kernel.

    ``flag_gems.ops.convert_indices_from_coo_to_csr`` calls the kernel through
    its module global, so rebinding the attribute keeps the change
    backend-local (generic source untouched, other vendors unaffected). The
    vendor ops package is imported by ``SpecOpRegistrar`` after
    ``flag_gems.ops`` has loaded, so the generic module is already in
    ``sys.modules``.
    """
    mod = sys.modules.get("flag_gems.ops.convert_indices_from_coo_to_csr")
    if mod is not None:
        mod._coo_to_csr_lower_bound_kernel = _coo_to_csr_lower_bound_kernel


_patch_generic_kernel()


__all__ = ["_coo_to_csr_lower_bound_kernel"]
