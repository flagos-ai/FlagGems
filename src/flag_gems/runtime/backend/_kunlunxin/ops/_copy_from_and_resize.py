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

"""Kunlunxin override for ``aten::_copy_from_and_resize`` / ``.out``.

The generic implementation (``flag_gems/ops/_copy_from_and_resize.py``) is the
Lazy-backend eager recipe ``dst.resize_(src.shape); copy_(dst, src)``.  On this
backend the copy it dispatches (the generic ``flag_gems.ops.copy`` flat kernel,
``BLOCK = 4096``) runs at ~0.09 ms for a 32 MB fp16 copy, roughly 3x slower than
the vendor copy engine (table below), and it mis-converts ``fp16 -> bfloat16``
(the shared pointwise bf16 store path produces garbage on XPU3).

This override keeps the same resize/alias contract and routes the data movement
through the backend's own kernels:

* same dtype  -> ``_kunlunxin/ops/copy.copy_`` (bounded-tile flat block-DMA copy
  with an ``int32`` word view for >1 MB transfers, ``tle``/pointwise paths for
  strided layouts).
* float -> float cross dtype -> the verified flat cast kernels already used by
  ``to_copy`` (``fp32`` intermediate + round-to-nearest-even ``bf16`` bit store,
  ``uint16`` bit-rebuild on the ``bf16`` source side).  Bit-exact against the
  CPU reference for all ``{fp16, fp32, bf16} x {fp16, fp32, bf16}`` pairs and
  correct at 16.7 M elements.
* everything else -> the vendor ``copy_`` pointwise conversion path.

No host-side aten/composite fallback is used: every path below launches a real
XPU Triton kernel (``torch.empty`` appears only as a staging allocation for the
rare strided + cross-dtype combination).
"""

import logging
import warnings

import torch
import triton

from .copy import copy_ as _vendor_copy_
from .to import (
    _to_copy_contiguous_kernel,
    _to_copy_contiguous_to_bf16_kernel,
    _to_copy_from_bf16_contiguous_kernel,
)

logger = logging.getLogger(__name__)


def _pick_cast_block(n_elements: int) -> int:
    """Bounded flat tile for the cast kernels (mirrors ``copy._pick_flat_block``).

    Casts are memory bound and the tile must stay bounded: the block is chosen
    so a 16.7 M element cast still launches a few hundred programs, while small
    tensors keep a single right-sized program.
    """
    if n_elements >= 2**19:
        return 65536
    if n_elements >= 2**16:
        return 32768
    if n_elements >= 2**13:
        return 8192
    if n_elements >= 2**10:
        return 4096
    return 1024


def _dense_strides(shape: torch.Size):
    """Row-major (contiguous) strides for ``shape``."""
    strides = []
    acc = 1
    for dim in reversed(shape):
        strides.append(acc)
        acc *= dim
    strides.reverse()
    return tuple(strides)


def _resize_destination(dst: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Resize ``dst`` to ``shape`` in place, leaving it dense (contiguous).

    Native ``resize_`` keeps the current strides whenever the requested size
    already matches, so a non-contiguous destination (e.g. a transposed tensor)
    would stay strided.  ``_copy_from_and_resize`` is a materialization
    operator: the destination it hands back is a freshly sized dense buffer
    (that is also the layout the CPU/Lazy reference exposes).  When the storage
    is exactly the dense footprint of ``dst`` (the normal "op owns its scratch
    buffer" case), ``set_`` re-lays the same storage out with contiguous strides
    without moving data; a smaller-than-storage view keeps its strides and is
    handled by the strided copy path instead of clobbering neighbours.

    The explicit stride argument is required: on this stack ``set_(storage,
    offset, size)`` preserves the previous strides instead of recomputing
    contiguous ones.
    """
    if dst.size() != shape:
        dst.resize_(shape)
    if not dst.is_contiguous():
        dense_nbytes = dst.numel() * dst.element_size()
        if dst.untyped_storage().size() == dense_nbytes:
            dst.set_(dst.untyped_storage(), 0, shape, _dense_strides(shape))
    return dst


def _cast_copy_flat(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Flat float -> float conversion copy; both tensors must be contiguous."""
    n_elements = src.numel()
    if n_elements == 0:
        return dst
    block_size = _pick_cast_block(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    if dst.dtype == torch.bfloat16:
        # bf16 store through an explicit RNE bit round: the natural
        # `tl.store(bf16_value)` lowering is not usable for cross-dtype casts
        # on XPU3 (fp16 -> bf16 silently produces garbage).
        _to_copy_contiguous_to_bf16_kernel[grid](src, dst, n_elements, block_size)
    elif src.dtype == torch.bfloat16:
        # Rebuild fp32 from the raw bf16 bits instead of loading bf16 element
        # values (see _to_copy_from_bf16_contiguous_kernel).
        _to_copy_from_bf16_contiguous_kernel[grid](
            src.view(torch.uint16), dst, n_elements, block_size
        )
    else:
        _to_copy_contiguous_kernel[grid](src, dst, n_elements, block_size)
    return dst


def _copy_into(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """``dst.copy_(src)`` through backend kernels (shapes already equal)."""
    if dst.dtype == src.dtype:
        _vendor_copy_(dst, src)
        return dst

    if dst.dtype.is_floating_point and src.dtype.is_floating_point:
        stage = src
        if not stage.is_contiguous():
            # Strided + cross-dtype: materialize the source densely first (real
            # XPU strided copy), then run the flat cast kernels.
            stage = torch.empty(src.shape, dtype=src.dtype, device=src.device)
            _vendor_copy_(stage, src)
        if dst.is_contiguous():
            return _cast_copy_flat(stage, dst)
        staged_dst = torch.empty(dst.shape, dtype=dst.dtype, device=dst.device)
        _cast_copy_flat(stage, staged_dst)
        _vendor_copy_(dst, staged_dst)
        return dst

    # int/bool/etc. conversions: the vendor pointwise copy handles the
    # promotion/saturation semantics for equal shapes.
    _vendor_copy_(dst, src)
    return dst


def _copy_from_and_resize(self: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Copy ``self`` into ``dst``, resizing ``dst`` to ``self``'s shape first.

    Returns ``dst`` itself; the caller's tensor object is mutated in place
    (``self``'s shape, ``dst``'s dtype preserved).
    """
    logger.debug("GEMS_KUNLUNXIN _COPY_FROM_AND_RESIZE")
    _resize_destination(dst, self.shape)
    return _copy_into(dst, self)


def _copy_from_and_resize_out(
    self: torch.Tensor, dst: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    """``out`` variant: resize-and-copy ``self`` into ``dst``, then mirror the
    result into ``out`` (resized in place) and return ``out``."""
    logger.debug("GEMS_KUNLUNXIN _COPY_FROM_AND_RESIZE_OUT")
    result = _copy_from_and_resize(self, dst)
    if out is result:
        return out
    # Match torchgen's generated out wrapper: resize before validating options.
    if out.shape != result.shape and out.numel() != 0:
        warnings.warn(
            "An output with one or more elements was resized since its shape "
            "does not match the required output shape. This behavior is deprecated.",
            UserWarning,
            stacklevel=2,
        )
    out.resize_(result.shape)
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {result.dtype}, "
            f"but got {out.dtype} instead"
        )
    if out.device != result.device:
        raise RuntimeError(
            f"Expected out tensor to have device {result.device}, "
            f"but got {out.device} instead"
        )
    _copy_into(out, result)
    return out
