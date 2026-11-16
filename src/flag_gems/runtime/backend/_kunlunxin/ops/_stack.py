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
"""Kunlunxin (XPU3) override for ``aten::_stack``.

The generic implementation (``flag_gems/ops/_stack.py``) is a per-output-element
kernel whose inner index is a *runtime* ``int64`` scalar:

* every element computes ``pre = idx // post`` and ``post_i = idx % post``
  (two soft integer divisions per lane, ~30 soft ops each on this backend),
* every address is formed from a lane-varying ``int64`` vector,
* loads/stores are masked even though the grid is exact,
* and the ``int``/``float`` payload is moved through 4-byte element lanes.

On XPU3 that lowers to ~4 GB/s (measured: ``(4096,4096)`` fp32 stack takes
111 ms vs 0.20 ms for ``torch._stack``), i.e. 0.002-0.006x.

This override keeps the exact operator contract (dim validation including the
single-input case, equal-shape check, dtype promotion, complex handling via a
real view, zero-sized inputs) but replaces the kernel with what the hardware
does well:

1. ``dim == 0``: a stack along a *new leading* axis is just "pack N contiguous
   slabs into a fresh buffer" -- bit-identical to ``flatten_dense_tensors``,
   which already has a tuned backend kernel here.  Delegate to it: one 2-D
   ``(slot, chunk)`` launch of maskless block DMA over the widest integer word
   view (``technique-xpu3-multi-input-concat-word-copy``).
2. ``dim != 0``: the writer would have to interleave columns (per-element
   scatter, ~1.8 s for 3x(4096,4096) on XPU3).  Instead build a contiguous
   ``(N, *shape)`` slab with the same block-DMA copies and return the
   shape-correct **permuted view**.  This is the same pattern the vendored
   ``stack`` / ``cat`` inner-dim path already uses; values are identical.

No CPU / ATen / native computation is used: ``torch.empty`` only allocates,
``view``/``permute`` are metadata, and the data move is a real XPU Triton
kernel (``_flatten_words_kernel`` / ``_flat_elem_kernel`` / ``_gather_kernel``
from the backend ``flatten_dense_tensors`` module).
"""

import logging
from typing import List, Tuple, Union

import torch

from .flatten_dense_tensors import (
    _MAX_RANK,
    _MAX_SLOTS,
    _TL_DTYPE,
    _copy_flat,
    _copy_strided,
    _launch_words,
    _word_view,
)

logger = logging.getLogger(__name__)


def _contiguous_strides(shape):
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    strides.reverse()
    return strides


def _fill_slabs(tensors, out_flat, numels, elem_offsets) -> None:
    """Copy each input into its contiguous slab of the flat output buffer.

    Fast path: all inputs contiguous, same dtype as the output and at most
    ``_MAX_SLOTS`` of them -> one 2-D word-copy launch for the whole list.
    Otherwise one launch per input (flat affine copy for contiguous inputs,
    bounded rank affine gather for strided inputs).
    """
    out_dtype = out_flat.dtype
    if (
        len(tensors) <= _MAX_SLOTS
        and all(t.dtype == out_dtype for t in tensors)
        and all(t.is_contiguous() for t in tensors)
    ):
        wdt, per_word = _word_view(
            out_flat, tensors, numels, elem_offsets, tensors[0].element_size()
        )
        if wdt is not None:
            _launch_words(out_flat, tensors, numels, elem_offsets, per_word, wdt)
            return

    for i, t in enumerate(tensors):
        if numels[i] == 0:
            continue
        tl_dt = _TL_DTYPE.get(t.dtype)
        if tl_dt is None:
            raise RuntimeError("_stack: unsupported dtype %s" % t.dtype)
        if t.is_contiguous():
            _copy_flat(out_flat, t, elem_offsets[i], tl_dt)
        elif len(t.shape) - sum(1 for s in t.shape if s == 1) > _MAX_RANK:
            # The gather kernel collapses unit dims; a rank above its bound is
            # materialised first (same device-side move the generic path does
            # with ``.contiguous()``).
            c = t.contiguous()
            _copy_flat(out_flat, c, elem_offsets[i], tl_dt)
        else:
            _copy_strided(out_flat, t, elem_offsets[i], tl_dt)


def _stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    # Validate ``dim`` against tensors[0] unconditionally: with a single input
    # the per-entry loop below never runs and an out-of-range dim would be
    # silently accepted instead of matching ATen.
    ndim0 = tensors[0].dim()
    if (dim < -ndim0 - 1) or (dim > ndim0):
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -ndim0 - 1, ndim0, dim
            )
        )
    for i, s in enumerate(inp_shapes[1:]):
        ndim = tensors[i + 1].dim()
        if (dim < -ndim - 1) or (dim > ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -ndim - 1, ndim, dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + ndim0 + 1

    # Type promotion: identical to the generic operator / ATen.
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)

    # Triton has no complex scalar types. aten::_stack accepts complex dtypes,
    # so stack the real view of every input and reinterpret at the end.
    is_complex = dtype.is_complex
    if is_complex:
        complex_dtype = dtype
        tensors = [
            torch.view_as_real(t if t.dtype == complex_dtype else t.to(complex_dtype))
            for t in tensors
        ]
        dtype = tensors[0].dtype
    else:
        tensors = [t if t.dtype == dtype else t.to(dtype) for t in tensors]

    # ``work_shape`` includes the trailing real dimension for complex inputs;
    # ``dim`` was normalised against the logical rank, so it always points
    # before (or at) that trailing axis.
    work_shape = list(tensors[0].shape)
    n = len(tensors)
    S = 1
    for s in work_shape:
        S *= s

    out_shape = work_shape[:dim] + [n] + work_shape[dim:]
    device = tensors[0].device
    # ``empty_strided`` with contiguous strides bypasses the GEMS
    # ``aten::empty.memory_format`` override, which is documented (generic
    # ``_stack``) to launch a Triton zero-init kernel; profiling on this card
    # showed a ``constant<int>`` kernel at ~0.12 ms per call, i.e. a redundant
    # full write of the output. Every element is written below.
    if S == 0:
        out = torch.empty_strided(
            out_shape, _contiguous_strides(out_shape), dtype=dtype, device=device
        )
        return torch.view_as_complex(out) if is_complex else out

    flat = torch.empty_strided((n * S,), (1,), dtype=dtype, device=device)
    numels = [S] * n
    elem_offsets = [i * S for i in range(n)]
    _fill_slabs(tensors, flat, numels, elem_offsets)

    if dim == 0:
        out = flat.view(out_shape)
    else:
        # ``flat`` is a contiguous (N, *shape) slab; the stacked axis lives on
        # axis 0, so a metadata permute yields the requested dim without any
        # column-interleaved scatter.
        k = len(work_shape) - 1
        perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, k + 2))
        out = flat.view([n] + work_shape).permute(perm)

    if is_complex:
        out = torch.view_as_complex(out)
    return out
