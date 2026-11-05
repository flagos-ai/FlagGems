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

"""Ascend copy: the shared implementation plus the three things it cannot do here.

Everything else is delegated to ``flag_gems.ops.copy.copy_`` unchanged.

1. The flat path launches ``cdiv(n_elements, 4096)`` programs directly, and it
   is the one launch in that file that does not go through ``pointwise_dynamic``
   -- so the grid bound this backend declares in ``CodeGenConfig`` never reaches
   it. The driver flattens the grid into a uint16 coreDim, and a copy of more
   than 65535 tiles is refused outright: a Qwen3.5-27B step dies in
   ``multi_tensor_scale_tensor`` with ``coreDim 77600`` on a 317849600-element
   gradient buffer. The kernel below bounds the grid and walks the rest of the
   tiles inside the program, so the flat path stays available at every size,
   which is worth about 2x over the pointwise path (median kernel us, bfloat16):

       elements     flat   pointwise   aclnnInplaceCopy
       2**22        19.7        26.7                7.4
       2**26       290.2       591.7              182.2
       2**28      1182.9      2354.9              846.2

2. ``pointwise_dynamic``'s generated code reads the wrong bytes from element 32
   on when a 1-byte operand is not contiguous -- int8, uint8 and bool all return
   wrong data, int16 and wider are fine. A hand-written kernel with the same
   strides is correct, so this is the codegen rather than the backend, and
   ``abs``/``neg``/``bitwise_not`` are wrong on the same inputs. Making the
   source contiguous first keeps copy correct while that is fixed elsewhere.

3. Every fallback in the shared implementation hands the case back with
   ``redispatch(CompositeExplicitAutograd)``, whose composite kernel calls
   ``aten::_copy_from``. torch_npu does not register it, falls back to CPU, and
   finds no CPU kernel either, so an empty tensor or a cross-device copy raises
   ``NotImplementedError`` instead of working. Those cases are resolved here
   with operators that are still native (``to``, ``clone``, ``resolve_neg``).
"""

import logging
import threading
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import _int32_viewable
from flag_gems.ops.copy import copy_ as _generic_copy_
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Set while this function is moving data between devices; see copy_.
_reentry = threading.local()

_current_device = getattr(torch_device_fn, "current_device", None)

# 4096 x int32 = 16 KB per program, well inside the 192 KB UB.
_FLAT_COPY_BLOCK = 4096
# The driver flattens the launch grid into a uint16 coreDim before
# rtKernelLaunch (coreDim = grid.x * grid.y * grid.z).
_MAX_CORE_DIM = 65535
# pointwise_dynamic computes its offsets in int32, so a copy wider than this is
# split rather than handed to it. Lowered in the tests to exercise the split.
_MAX_POINTWISE_ELEMS = 2**31 - 1


@triton.jit
def _copy_tile(src_ptr, dst_ptr, offsets, n_elements, NEED_MASK: tl.constexpr):
    if NEED_MASK:
        mask = offsets < n_elements
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


@triton.jit
def _flat_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    """Flat copy over a grid the driver can express.

    The tile count may exceed the grid, so each program walks its tiles with a
    stride of the grid width. ``NEED_MASK`` keeps the exact-multiple case off
    the masked path. ``WIDE_OFFSETS`` widens ``block_id`` before it is scaled:
    once the grid is bounded, ``block_id`` runs to the tile count rather than to
    the grid, and ``block_id * BLOCK_SIZE`` leaves int32 before the element
    count does -- but only past 2**31 elements, and int32 offsets are cheaper,
    so the launch picks.
    """
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        if WIDE_OFFSETS:
            offsets = block_id.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            _copy_tile(src_ptr, dst_ptr, offsets, n_elements, NEED_MASK)
        else:
            offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            _copy_tile(src_ptr, dst_ptr, offsets, n_elements, NEED_MASK)


@triton.jit
def _gather_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Read a strided source by its own strides and write it out contiguously.

    pointwise_dynamic mis-reads an operand narrower than 4 bytes once it is not
    contiguous, and materialising the source with ``contiguous()`` is not an
    option: that re-enters ``aten::copy_``, which is this function, so it
    recurses as soon as copy_ is registered for dispatch.
    """
    for block_id in range(
        tl.program_id(0), tl.cdiv(n_elements, BLOCK_SIZE), tl.num_programs(0)
    ):
        index = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = index < n_elements
        remaining = index
        offsets = index * 0
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            offsets += (remaining % SHAPE[dim]) * STRIDES[dim]
            remaining = remaining // SHAPE[dim]
        tl.store(dst_ptr + index, tl.load(src_ptr + offsets, mask=mask), mask=mask)


def _gather_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst`` contiguous, ``src`` strided, same shape and dtype."""
    n_elements = src.numel()
    _gather_copy_kernel[_flat_grid(n_elements)](
        src,
        dst.reshape(-1),
        n_elements,
        tuple(src.shape),
        tuple(src.stride()),
        _FLAT_COPY_BLOCK,
    )


def _flat_plan(dst: torch.Tensor, src: torch.Tensor):
    """``(as_int32, n_elements)`` for the flat kernel, or None.

    Same shape, same dtype, both contiguous. Unlike the shared plan there is no
    element ceiling: the grid is bounded by the launch instead of by falling
    back, and the offsets are int64.
    """
    if src.shape != dst.shape or src.dtype != dst.dtype:
        return None
    if not src.is_contiguous() or not dst.is_contiguous():
        return None
    as_int32 = _int32_viewable(src) and _int32_viewable(dst)
    n_elements = src.numel()
    if as_int32:
        n_elements = n_elements * src.element_size() // 4
    return as_int32, n_elements


def _flat_grid(n_elements: int):
    """The launch grid for ``n_elements``, bounded to what the driver can express."""
    return (min(triton.cdiv(n_elements, _FLAT_COPY_BLOCK), _MAX_CORE_DIM),)


def _flat_copy(dst: torch.Tensor, src: torch.Tensor, as_int32: bool, n_elements: int):
    view = src.reshape(-1)
    out = dst.reshape(-1)
    if as_int32:
        view = view.view(torch.int32)
        out = out.view(torch.int32)

    grid = _flat_grid(n_elements)
    need_mask = n_elements % _FLAT_COPY_BLOCK != 0
    # The largest offset a tile forms is below n_elements + BLOCK; past that the
    # product leaves int32 and the mask would be applied to a wrapped value.
    wide = n_elements + _FLAT_COPY_BLOCK > 2**31 - 1
    device_index = view.device.index
    if (
        _current_device is None
        or device_index is None
        or _current_device() == device_index
    ):
        _flat_copy_kernel[grid](
            view, out, n_elements, _FLAT_COPY_BLOCK, need_mask, wide
        )
    else:
        with torch_device_fn.device(device_index):
            _flat_copy_kernel[grid](
                view, out, n_elements, _FLAT_COPY_BLOCK, need_mask, wide
            )


def _check_broadcastable(dst: torch.Tensor, src: torch.Tensor) -> None:
    """What PyTorch validates before it copies, including for empty tensors."""
    if src.shape == dst.shape:
        return
    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc
    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if isinstance(src, (int, float, bool)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("unsupport src type for copy_: ", type(src))

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    # Layouts and quantization are the shared implementation's to refuse; taking
    # them over here would be a different answer, not a fix.
    if (
        dst.layout != torch.strided
        or src.layout != torch.strided
        or dst.is_quantized
        or src.is_quantized
    ):
        return _generic_copy_(dst, src, non_blocking)

    # Materialise lazy math bits rather than handing them back: the fallback
    # that would resolve them is the one that does not work here.
    if src.is_neg() or src.is_conj():
        src = src.resolve_neg().resolve_conj()

    if src.device != dst.device:
        # A transfer needs the runtime, not a kernel, and every primitive that
        # could do it -- to, _to_copy, Storage.copy_ -- routes back through
        # aten::copy_. That is fine while copy_ is not registered for dispatch,
        # which is the case on Ascend (it is in CUSTOMIZED_UNUSED_OPS), and it
        # recurses forever as soon as it is. Say so instead of overflowing.
        if getattr(_reentry, "transferring", False):
            raise NotImplementedError(
                "copy_ cannot move data between devices while it is registered "
                "for dispatch: every transfer primitive routes back through "
                "aten::copy_. Leave copy_ in CUSTOMIZED_UNUSED_OPS."
            )
        _reentry.transferring = True
        try:
            src = src.to(dst.device)
        finally:
            _reentry.transferring = False
        if dst.device.type == "cpu":
            return dst.copy_(src, non_blocking=non_blocking)

    if torch._C._is_alias_of(dst, src):
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        # A temporary is what PyTorch uses to make overlapping writes defined.
        src = src.clone()

    if dst.numel() == 0:
        _check_broadcastable(dst, src)
        return dst

    # pointwise_dynamic mis-reads a strided operand narrower than 4 bytes, so
    # read it here instead. Writing straight into a contiguous destination is
    # the whole copy; otherwise this only materialises the source.
    if src.element_size() < 4 and src.ndim > 0 and not src.is_contiguous():
        if dst.is_contiguous() and src.dtype == dst.dtype and src.shape == dst.shape:
            logger.debug("GEMS COPY_")
            _gather_copy(dst, src)
            return dst
        contiguous_src = torch.empty_like(src, memory_format=torch.contiguous_format)
        _gather_copy(contiguous_src, src)
        src = contiguous_src

    plan = _flat_plan(dst, src)
    if plan is not None:
        logger.debug("GEMS COPY_")
        _flat_copy(dst, src, *plan)
        return dst

    if dst.numel() > _MAX_POINTWISE_ELEMS:
        # The flat path above already takes any size; what is left here would go
        # to pointwise_dynamic, whose offsets are int32. Halving is exact -- each
        # half is a copy_ in its own right -- and recurses until it fits.
        if dst.ndim > 0 and dst.shape[0] > 1:
            expanded = src if src.shape == dst.shape else src.expand(dst.shape)
            half = dst.shape[0] // 2
            copy_(dst[:half], expanded[:half], non_blocking)
            copy_(dst[half:], expanded[half:], non_blocking)
            return dst
        raise NotImplementedError(
            f"copy_ on Ascend cannot split a {tuple(dst.shape)} destination of "
            f"{dst.numel()} elements into pieces pointwise_dynamic can index"
        )

    try:
        return _generic_copy_(dst, src, non_blocking)
    except NotImplementedError as exc:
        if "_copy_from" not in str(exc):
            raise
        # The shared implementation hands the case back with
        # redispatch(CompositeExplicitAutograd), whose composite kernel calls
        # aten::_copy_from. torch_npu does not register it, so the error that
        # comes out names an operator the caller never asked for.
        raise NotImplementedError(
            "copy_ cannot serve this case on Ascend: the shared implementation "
            "returns it to PyTorch, and that path needs aten::_copy_from, which "
            f"torch_npu does not register. Underlying error: {exc}"
        ) from exc


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out
