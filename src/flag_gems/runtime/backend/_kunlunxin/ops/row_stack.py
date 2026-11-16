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
"""Kunlunxin (XPU3) override for ``aten::row_stack`` / ``aten::row_stack.out``.

``row_stack`` is an alias of ``vstack``: every input is lifted to at least 2-D
(``atleast_2d``) and the result is the ``cat`` along dim 0.  The generic
implementation copies it with a ``vstack``-autotuned row-block kernel whose
grid/staging path measures ~0.15x on the large benchmark shapes (0.70 ms for a
3x(4096,4096) stack vs 0.10 ms for ``torch``).

On XPU3 a dim-0 concatenation is just "pack every input, in vertical order,
into one contiguous buffer".  For contiguous inputs that is bit-identical to
``flatten_dense_tensors``, which already has a tuned backend kernel here (2-D
``(slot, chunk)`` maskless block DMA over the widest same-dtype integer word
view).  This override reuses that machinery (``_fill_slabs``/``_fill`` from the
backend ``_stack`` module):

* contiguous inputs -> a single word-copy launch,
* other layouts (transpose / channels-last / strided) -> the bounded-rank
  affine gather kernel, which maps the *logical* row-major index onto the
  input strides and therefore produces the same contiguous values as ATen,
* mixed dtypes -> type promotion followed by an element copy,
* complex -> stack the real view, reinterpret at the end.

Output layout matches this vendor's ATen: the functional result is contiguous.
For ``row_stack.out`` the supplied ``out`` is kept when its shape already
matches (even channels-last / strided), and a wrong-shaped ``out`` is
``resize_``-d to a contiguous buffer -- exactly the ATen behaviour measured on
this device (a resized ``cat_out`` does *not* re-impose a derived memory
format).

No CPU / ATen / native computation is used for the copy: ``torch.empty_strided``
only allocates, ``atleast_2d``/``view`` are metadata, and the data movement is
the backend Triton kernels.  ``out.copy_(staging)`` on the rare non-contiguous
out path is a native device-side data move (never ``aten._copy_from``).
"""

import logging

import torch

from ._stack import _fill_slabs

logger = logging.getLogger(__name__)


def _normalize(tensors):
    """``atleast_2d`` plus vstack's validation (same diagnostics as generic)."""
    tensors = list(torch.atleast_2d(tensors))
    if len(tensors) == 0:
        raise RuntimeError("vstack expects a non-empty TensorList")

    device = tensors[0].device
    ndim = tensors[0].dim()
    for i, tensor in enumerate(tensors[1:], start=1):
        if tensor.device != device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got "
                f"{tensor.device} and {device}"
            )
        if tensor.dim() != ndim:
            raise RuntimeError(
                "Tensors must have same number of dimensions: got "
                f"{ndim} and {tensor.dim()}"
            )
        for dim, (expected, got) in enumerate(zip(tensors[0].shape, tensor.shape)):
            if dim != 0 and expected != got:
                raise RuntimeError(
                    "Sizes of tensors must match except in dimension 0. "
                    f"Expected size {expected} but got size {got} for tensor "
                    f"number {i} in the list."
                )
    return tensors


def _promote_dtype(tensors):
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        if tensor.dtype != dtype:
            dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _prepare(tensors):
    tensors = _normalize(tensors)
    dtype = _promote_dtype(tensors)
    out_shape = list(tensors[0].shape)
    out_shape[0] = sum(t.shape[0] for t in tensors)
    return tensors, dtype, out_shape


def _slabs(tensors):
    """Per-input slab plan: drop empty inputs and use their real ``numel()``.

    Each input keeps its own (possibly different) dim-0 extent, so the packed
    slab size is ``t.numel()`` -- never the per-row width.  Empty inputs are
    short-circuited here so no kernel ever sees a zero-length slab.
    """
    kept = []
    numels = []
    offsets = []
    total = 0
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        kept.append(t)
        numels.append(n)
        offsets.append(total)
        total += n
    return kept, numels, offsets, total


def _run(tensors, dtype, out_shape):
    """Allocate a contiguous output and pack every input into it."""
    device = tensors[0].device
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

    kept, numels, offsets, total = _slabs(tensors)

    if total == 0:
        # No data to move.  ``torch.empty`` yields ATen's default strides,
        # which treat a size-0 dim as size 1 when accumulating (e.g. a (5, 0)
        # output has strides (1, 1), not (0, 1)); ``_contiguous_strides`` with
        # a plain running product would get that wrong.
        if is_complex:
            return torch.empty(out_shape, dtype=complex_dtype, device=device)
        return torch.empty(out_shape, dtype=dtype, device=device)

    work_out_shape = list(out_shape) + ([2] if is_complex else [])
    flat = torch.empty_strided((total,), (1,), dtype=dtype, device=device)
    _fill_slabs(kept, flat, numels, offsets)
    out = flat.view(work_out_shape)
    return torch.view_as_complex(out) if is_complex else out


def _check_out(out, out_shape, dtype, tensors):
    """ATen ``cat_out`` validation: castability, resize, overlap checks."""
    if not torch.can_cast(dtype, out.dtype):
        raise TypeError(
            "torch.cat(): input types can't be cast to the desired output type "
            f"{out.dtype}"
        )

    if list(out.shape) != list(out_shape):
        # ``resize_`` installs contiguous strides.  This mirrors ATen on this
        # device: a resized ``cat_out`` target ends up contiguous rather than
        # re-imposing a derived channels-last format.
        out.resize_(out_shape)

    from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

    if has_internal_overlapping(out) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    for tensor in tensors:
        if torch._C._overlaps(out, tensor):
            raise RuntimeError(
                "unsupported operation: some elements of the input tensor and "
                "the written-to tensor refer to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
    return out


def _fill_into(out, tensors, dtype):
    """Write the stacked values into ``out`` (any layout, matching dtype)."""
    kept, numels, offsets, total = _slabs(tensors)
    if total == 0:
        return

    # Triton has no complex scalar type: drive the kernels on the real view of a
    # complex destination, mirroring the input handling in ``_run``.
    target = torch.view_as_real(out) if out.is_complex() else out
    if target.is_contiguous():
        _fill_slabs(kept, target.view(-1), numels, offsets)
    else:
        staging = torch.empty_strided((total,), (1,), dtype=dtype, device=out.device)
        _fill_slabs(kept, staging, numels, offsets)
        target.copy_(staging.view(target.shape))


def row_stack(tensors):
    logger.debug("GEMS_KUNLUNXIN ROW_STACK")
    tensors, dtype, out_shape = _prepare(tensors)
    return _run(tensors, dtype, out_shape)


def row_stack_out(tensors, *, out):
    logger.debug("GEMS_KUNLUNXIN ROW_STACK_OUT")
    tensors, dtype, out_shape = _prepare(tensors)
    _check_out(out, out_shape, dtype, tensors)

    if out.numel() == 0:
        return out

    if dtype.is_complex:
        complex_dtype = dtype
        tensors = [
            torch.view_as_real(t if t.dtype == complex_dtype else t.to(complex_dtype))
            for t in tensors
        ]
        dtype = tensors[0].dtype
    else:
        tensors = [t if t.dtype == dtype else t.to(dtype) for t in tensors]

    _fill_into(out, tensors, dtype)
    return out
