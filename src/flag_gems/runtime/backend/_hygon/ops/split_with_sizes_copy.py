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

import logging

import torch

logger = logging.getLogger(__name__)


def _normalize_split_sizes(split_sizes):
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()

    if hasattr(split_sizes, "__iter__"):
        split_sizes = list(split_sizes)

    return [int(size) for size in split_sizes]


def _normalize_dim(inp, dim):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    return dim % inp.ndim


def split_with_sizes_copy(inp, split_sizes, dim=0):
    """Hygon specialization of ``split_with_sizes_copy``.

    The generic Triton implementation is launch-overhead bound: it materializes
    a full clone of the input, runs an indexed-copy kernel, and does Python-side
    bookkeeping, so it ran at 0.05-0.33x of eager on every benchmarked shape
    while passing all accuracy tests.

    The op has no numerical work at all -- it is a pure gather-and-copy. The
    fast path is therefore to stay in ATen and copy the split views in bulk.
    Two subtleties matter because this runs inside ``flag_gems.use_gems()``:

    * ``Tensor.copy_`` is patched with a slow generic gems kernel (~0.15 ms for
      a tiny copy vs ~0.008 ms native). ``torch._foreach_copy_`` is an unpatched
      bulk-copy primitive that is ~20x faster, so all splits are copied in one
      fused foreach call.
    * ``torch.split_with_sizes`` / ``narrow`` / ``slice`` are gems-registered and
      re-enter Triton. ``torch.as_strided`` is unpatched, so the source split
      views are constructed directly with it.

    ``clone`` decomposes to ``copy_`` and is therefore also slow -- it is avoided
    in favor of ``new_empty`` (cheap inside use_gems) plus a foreach copy.
    """
    logger.debug("GEMS_HYGON SPLIT_WITH_SIZES_COPY")

    dim = _normalize_dim(inp, dim)
    split_sizes = _normalize_split_sizes(split_sizes)

    split_sum = sum(split_sizes)
    assert split_sum == inp.shape[dim], "Invalid split_sizes"

    in_shape = list(inp.shape)
    in_strides = list(inp.stride())

    # Build source views for each split via as_strided (no kernel, no re-entry
    # into the patched split/narrow ops).
    srcs = []
    dim_start = 0
    for size in split_sizes:
        out_shape = list(in_shape)
        out_shape[dim] = size
        offset = inp.storage_offset() + dim_start * in_strides[dim]
        srcs.append(torch.as_strided(inp, out_shape, in_strides, offset))
        dim_start += size

    # dim==0 && contiguous: a single contiguous output buffer lets us copy the
    # whole input in one shot and hand back strided views. One allocation beats
    # per-split allocation on large shapes.
    if dim == 0 and inp.is_contiguous():
        buf = inp.new_empty(in_shape)
        torch._foreach_copy_([buf], [inp])

        result = []
        elem_offset = 0
        for size in split_sizes:
            out_shape = list(in_shape)
            out_shape[dim] = size
            result.append(torch.as_strided(buf, out_shape, buf.stride(), elem_offset))
            elem_offset += size * buf.stride(0)
        return tuple(result)

    # General case: contiguous per-split output, all copies fused into one
    # foreach_copy_ call.
    dsts = []
    for size in split_sizes:
        out_shape = list(in_shape)
        out_shape[dim] = size
        dsts.append(inp.new_empty(out_shape))

    if dsts:
        torch._foreach_copy_(dsts, srcs)

    return tuple(dsts)
