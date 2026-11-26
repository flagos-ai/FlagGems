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

"""Shared fast path for the *unmasked direct-pointer* foreach overrides.

The first foreach overrides on this backend (``_foreach_sigmoid_`` and its
siblings) packed the tensor pointers into a device-side metadata buffer and let
each program reload its pointer with ``tl.load(meta_ptr + t)``.  That works,
but for the memory-bound unary ops (``abs``/``sign``/``round``) it halves the
achievable bandwidth: the pointer arrives as a runtime global-memory load, so
the backend cannot apply its block-DMA lowering to ``tl.load(ptr + idx)`` even
though the address is dense and aligned (measured ~840 GB/s vs ~1780 GB/s for a
direct pointer argument on a 16.7M-element fp16 list).

Here the tensor pointers are passed *directly* as scalar kernel arguments and
selected with a ``tl.where`` chain over ``tl.program_id(0)``.  A scalar
``tl.where`` result cast back to ``tl.pointer_type`` stays compile-time
analysable, so block-DMA applies and memory-bound ops run at full bandwidth.
The chain has ``MAX_DIRECT = 16`` slots (the benchmark list length); shorter
lists pad the unused slots with the first tensor's pointer, longer lists fall
back to the generic ``foreach_unary`` executor (correctness only -- no in-place
benchmark or functional case exceeds 16 tensors).
"""

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import has_internal_overlap, is_flat_addressable, tl_dtype

# Number of scalar pointer slots in the direct-pointer kernel.  Matches
# ``ForeachUnaryBenchmark.LIST_LENGTH`` (16); longer lists take the fallback.
MAX_DIRECT = 16

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _select_ptr(
    t, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15
):
    """Return the ``t``-th pointer (int64 scalar) for the 2D-grid kernel."""
    return tl.where(
        t == 0,
        p0,
        tl.where(
            t == 1,
            p1,
            tl.where(
                t == 2,
                p2,
                tl.where(
                    t == 3,
                    p3,
                    tl.where(
                        t == 4,
                        p4,
                        tl.where(
                            t == 5,
                            p5,
                            tl.where(
                                t == 6,
                                p6,
                                tl.where(
                                    t == 7,
                                    p7,
                                    tl.where(
                                        t == 8,
                                        p8,
                                        tl.where(
                                            t == 9,
                                            p9,
                                            tl.where(
                                                t == 10,
                                                p10,
                                                tl.where(
                                                    t == 11,
                                                    p11,
                                                    tl.where(
                                                        t == 12,
                                                        p12,
                                                        tl.where(
                                                            t == 13,
                                                            p13,
                                                            tl.where(t == 14, p14, p15),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _pick_block(n_elements):
    """``(block_size, num_warps)`` ladder shared by the unmasked kernels.

    Large blocks amortize the per-program pointer selection and let the backend
    vectorize the access; the top rung is the winning 131072/32w recipe from the
    on-device sweep in ``_foreach_sigmoid_``.  Every block is a power of two, so
    the caller only takes the fast path when ``n_elements % block == 0``.
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


def _launch_fast(tensors, kernel):
    """Launch the direct-pointer ``kernel`` over a homogeneous ``tensors`` list.

    Returns ``True`` when the fast path ran; ``False`` otherwise (the caller
    falls back to ``foreach_unary``).  The precondition gate mirrors the shared
    executor: one device, one fast float dtype, dense/flat-addressable, no
    internal overlap, equal element count, at least 128 elements.
    """
    if not tensors or len(tensors) > MAX_DIRECT:
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
        if t.dtype != dtype or t.device != device or t.numel() != n_elements:
            return False
        # In-place safety and flat addressing are the same preconditions the
        # shared executor enforces; check them here too so an overlap/aliasing
        # write never slips past the fast path.
        if not is_flat_addressable(t):
            return False
        if has_internal_overlap(t):
            return False
    block, num_warps = _pick_block(n_elements)
    if n_elements % block != 0:
        return False
    ptrs = [t.data_ptr() for t in tensors]
    ptrs += [ptrs[0]] * (MAX_DIRECT - len(ptrs))
    grid = (len(tensors), n_elements // block)
    kernel[grid](
        *ptrs,
        tl_dtype(dtype),
        block,
        num_warps=num_warps,
        unroll_num=16,
        buffer_size_limit=8192,
        isCloseMemoryAsync=False,
    )
    return True
