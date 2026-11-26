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

"""XPU override for ``aten::_foreach_copy`` (out-of-place copy over Tensor[]).

``self`` and ``src`` are two ``Tensor[]`` of equal length; position ``i`` is
``out[i] = src[i]`` cast into ``self[i]``'s dtype (assignment semantics, no type
promotion).  The shared executor in :mod:`flag_gems.utils.foreach` handles the
generic case but its ``foreach_binary_kernel`` guards every load/store with a
runtime ``idx < n_elements`` mask, which blocks the block-DMA lowering even
when the mask is always-true (KERNEL_OPT_EXPERIENCE.md #36); for 16 x
(4096, 4096) that measures ~164 ms against ~1.25-2.4 ms for the torch
reference, and the per-call metadata H2D is the second-largest cost.

A *homogeneous* list -- one device, one src dtype, one dst dtype, every pair
contiguous (so both storage order and logical order coincide), all with the
same power-of-two element count -- is therefore processed by a dedicated
*unmasked* 2D-grid kernel (axis 0 selects the tensor, axis 1 the chunk) in a
single launch mirroring the ``_foreach_sigmoid`` fast path.  ``_copy_fn`` is a
pure ``return b``, so the kernel only ever touches the source pointer; the
destination is a freshly allocated ``empty_like`` output and the source is
cast on the way out.

Everything else -- different shapes/numels, non-contiguous or gappy views
(which the paired index math cannot flatten), internal overlaps, empty or
sub-block lists, mismatched lengths -- falls back to the generic
``_foreach_binary`` implementation, so validation and error messages stay
identical to the generic path.  The in-place ``_foreach_copy_`` is not
overridden here (it keeps the generic executor) and is unaffected.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _foreach_copy as _generic_foreach_copy
from flag_gems.utils.foreach import check_tensor_list, tl_dtype

logger = logging.getLogger(__name__)


@triton.jit
def _foreach_copy_unmasked_kernel(
    meta_ptr,
    NT: tl.constexpr,
    SRC_DT: tl.constexpr,
    DST_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over ``NT`` equal-size tensor pairs (out-of-place).

    ``meta_ptr`` holds the ``NT`` source pointers followed by the ``NT``
    destination pointers.  The grid is ``(NT, n_elements // BLOCK)``: axis 0
    selects the tensor, axis 1 the chunk inside it.  No mask anywhere, which is
    what lets the backend lower the accesses to block-DMA.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    src_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(SRC_DT))
    dst_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DST_DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src_ptr + idx)
    tl.store(dst_ptr + idx, x.to(DST_DT))


def _pick_block(n_elements, dtype):
    """``(block_size, num_warps)``, mirroring the sibling ``_foreach_sigmoid``.

    Every returned block is a power of two, so the caller only takes the fast
    path when ``n_elements % block == 0``.  ``dtype`` is accepted for
    signature parity with the sibling (a pure copy is dtype-agnostic on the
    block size).
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


def _launch_fast(tensors, srcs, dtype, device):
    n_elements = tensors[0].numel()
    block, num_warps = _pick_block(n_elements, dtype)
    if n_elements % block != 0:
        return None
    # Fresh outputs (pure allocation); ``empty_like`` keeps a contiguous
    # destination contiguous, matching the shared executor's staging path.
    outs = [torch.empty_like(t) for t in tensors]
    # One flat Python list -> one tensor -> one H2D transfer: source pointers
    # then destination pointers, both indexed off the same base.
    meta = torch.tensor(
        [t.data_ptr() for t in srcs] + [o.data_ptr() for o in outs],
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (len(tensors), n_elements // block)
    _foreach_copy_unmasked_kernel[grid](
        meta,
        len(tensors),
        tl_dtype(srcs[0].dtype),
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return outs


def _try_fast(tensors, srcs):
    if not tensors:
        return None
    first = tensors[0]
    dtype = first.dtype
    src_first = srcs[0]
    src_dtype = src_first.dtype
    device = first.device
    n_elements = first.numel()
    if n_elements < 1:
        return None
    for a, b in zip(tensors, srcs):
        if a.dtype != dtype:
            return None
        if b.dtype != src_dtype:
            return None
        if a.device != device or b.device != device:
            return None
        if a.numel() != n_elements or b.numel() != n_elements:
            return None
        # The flat copy maps storage order of the source onto storage order of
        # the destination; that is element-wise only when both are contiguous
        # (a transposed-but-dense view would silently transpose ``out``).
        if not a.is_contiguous() or not b.is_contiguous():
            return None
    return _launch_fast(tensors, srcs, dtype, device)


def _foreach_copy(self, src, non_blocking=False):
    del non_blocking
    logger.debug("GEMS_KUNLUNXIN _FOREACH_COPY")
    tensors = check_tensor_list(self)
    srcs = check_tensor_list(src)
    if len(tensors) != len(srcs):
        raise RuntimeError(
            f"Tensor list must have same number of elements as tensor list, "
            f"got {len(tensors)} and {len(srcs)}"
        )
    outs = _try_fast(tensors, srcs)
    if outs is not None:
        return outs
    return _generic_foreach_copy(tensors, srcs)


__all__ = ["_foreach_copy"]
