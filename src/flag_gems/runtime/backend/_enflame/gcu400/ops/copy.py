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
from typing import Optional

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

# 4096 int32 (16 KB) per program; the grid is capped so the loop is a
# grid-stride one instead of one CTA per tile.
_FLAT_COPY_BLOCK = 4096
_FLAT_COPY_GRID = 48
# Offsets are int32, so the last masked tile must still fit in an int32.
_FLAT_COPY_MAX_ELEMS = 2**31 - 1 - _FLAT_COPY_BLOCK


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


@triton.jit
def _flat_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Flat grid-stride copy over int32 words.

    The nd-tile pointwise kernel above moves every tile, tail included, through
    a block-pointer (DTE) transfer.  On gcu400 the DMA length of the last tile
    is rounded up to the tile width instead of stopping at the logical end, so
    a copy whose allocation happens to end on a page boundary faults the MMU
    (observed as ``LIBRA_MMU_INST_SIC_GRP receive page fault ras`` followed by
    a failed FLR that takes the card offline).  Plain offset loads/stores stay
    on the synchronous port and are masked exactly, so they cannot overrun.
    """
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(src_ptr + offsets, mask=mask, other=0)
        tl.store(dst_ptr + offsets, value, mask=mask)


def _int32_viewable(tensor: torch.Tensor) -> bool:
    """Whether ``tensor.reshape(-1).view(torch.int32)`` is expressible.

    Reinterpreting both operands as int32 keeps the kernel dtype-agnostic
    (fp8 included) and moves 128 bits per lane.
    """
    esize = tensor.element_size()
    if (tensor.numel() * esize) % 4 != 0:
        return False
    return (tensor.storage_offset() * esize) % 4 == 0


def _flat_copy_plan(dst: torch.Tensor, src: torch.Tensor):
    """Return ``(n_elements, word_dtype)`` for ``_flat_copy_kernel``, or None."""
    if src.shape != dst.shape or src.dtype != dst.dtype:
        return None
    if not src.is_contiguous() or not dst.is_contiguous():
        return None
    if _int32_viewable(src) and _int32_viewable(dst):
        word_dtype = torch.int32
        n_elements = src.numel() * src.element_size() // 4
    elif src.element_size() == 1:
        # Same-itemsize view, always expressible.  This also keeps
        # float8_e8m0fnu away from Triton's dtype mapping.
        word_dtype = torch.uint8
        n_elements = src.numel()
    else:
        return None
    if n_elements > _FLAT_COPY_MAX_ELEMS:
        return None
    return n_elements, word_dtype


def _flat_copy(dst: torch.Tensor, src: torch.Tensor, plan) -> bool:
    n_elements, word_dtype = plan
    try:
        view = src.reshape(-1).view(word_dtype)
        out = dst.reshape(-1).view(word_dtype)
    except (RuntimeError, TypeError, NotImplementedError):
        return False
    grid = (min(_FLAT_COPY_GRID, triton.cdiv(n_elements, _FLAT_COPY_BLOCK)),)
    _flat_copy_kernel[grid](
        view,
        out,
        n_elements,
        BLOCK_SIZE=_FLAT_COPY_BLOCK,
        GRID_SIZE=_FLAT_COPY_GRID,
        num_warps=4,
    )
    return True


def _copy_e8m0(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Handle float8_e8m0fnu copies that torch_gcu cannot convert.

    Two cases are expressible without a Triton float8 dtype: a same-dtype copy
    is a raw byte move, and ``e8m0 -> float`` is a pure exponent decode.
    """
    if src.shape != dst.shape:
        return False
    if src.dtype == dst.dtype:
        if dst.numel() == 0:
            return True
        if not src.is_contiguous() or not dst.is_contiguous():
            return False
        plan = _flat_copy_plan(dst, src)
        return plan is not None and _flat_copy(dst, src, plan)
    if (
        src.dtype == _FLOAT8_E8M0FNU
        and dst.is_floating_point()
        and dst.dtype != _FLOAT8_E8M0FNU
    ):
        if dst.numel() == 0:
            return True
        if not src.is_contiguous() or not dst.is_contiguous():
            return False
        n_elements = src.numel()
        if n_elements > _FLAT_COPY_MAX_ELEMS:
            return False
        try:
            view = src.reshape(-1).view(torch.uint8)
        except (RuntimeError, TypeError, NotImplementedError):
            return False
        grid = (min(_FLAT_COPY_GRID, triton.cdiv(n_elements, _FLAT_COPY_BLOCK)),)
        _e8m0_to_float_kernel[grid](
            view,
            dst.reshape(-1),
            n_elements,
            BLOCK_SIZE=_FLAT_COPY_BLOCK,
            GRID_SIZE=_FLAT_COPY_GRID,
            num_warps=4,
        )
        return True
    return False


@triton.jit
def _e8m0_to_float_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Decode float8_e8m0fnu (exponent only) to a float destination.

    e8m0 holds no mantissa: ``value = 2 ** (bits - 127)``, with 0xFF reserved
    for NaN and 0x00 mapping to 2**-127.  That is exactly the fp32 bit pattern
    ``bits << 23`` (the two special codes are patched in), which stays bit
    exact - ``tl.exp2`` is only accurate to ~1e-6.  torch_gcu cannot convert
    the dtype natively (``Cannot convert ScalarType Float8_e8m0fnu``), so the
    backend decodes it here from the uint8 view of the source.
    """
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        bits = tl.load(src_ptr + offsets, mask=mask, other=127).to(tl.int32)
        value_bits = tl.where(bits == 0, 1 << 22, bits << 23)
        value_bits = tl.where(bits == 255, 0x7FC00000, value_bits)
        value = value_bits.to(tl.float32, bitcast=True)
        tl.store(dst_ptr + offsets, value.to(dst_ptr.dtype.element_ty), mask=mask)


def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        # Preserve PyTorch's behaviour of warning when casting complex to real
        # by forcing the redispatch path, which issues the warning internally.
        return False
    if dst.dim() > 0 and dst.size(-1) > 1 and dst.stride(-1) != 1:
        # The strided pointwise store mis-addresses the tail when the
        # innermost dimension is not unit-stride: measured on a transposed
        # (7, 19) destination and on a channels_last 4-D destination, roughly
        # half of the elements received wrong values (sliced destinations with
        # stride(-1) == 1 are exact).  Reading a non-contiguous source is fine.
        return False
    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        # Triton does not support float8 yet, so defer to PyTorch which has a reference implementation.
        return False
    return True


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS_ENFLAME COPY")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


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

    if torch._C._is_alias_of(dst, src):
        # Align with PyTorch: if metadata fully matches, this is a no-op.
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.device == src.device
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        # Otherwise defer to PyTorch for well-defined semantics on overlapping writes.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        # torch_gcu cannot convert this dtype ("Cannot convert ScalarType
        # Float8_e8m0fnu to topsatenDataType_t"), so the backend handles the
        # cases Triton can express; anything else still goes to native.
        if _copy_e8m0(dst, src):
            return dst
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if src.numel() > 2**31 - 1 or dst.numel() > 2**31 - 1:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if not _can_use_triton(dst, src):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if dst.numel() == 0:
        # Respect PyTorch behaviour: empty tensors should still validate broadcast.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    logger.debug("GEMS_ENFLAME COPY_")

    try:
        broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if torch.Size(broadcast_shape) != dst.shape:
        raise RuntimeError(
            f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
        )

    # Contiguous same-dtype copies (the common case, and the only one that can
    # be large) bypass the DTE path; everything else keeps the pointwise kernel.
    plan = _flat_copy_plan(dst, src)
    if plan is not None and _flat_copy(dst, src, plan):
        return dst

    expanded_src = _expand_like(src, dst.shape)

    overload = _copy_kernel.instantiate(expanded_src.ndim)
    overload(expanded_src, out0=dst)
    return dst
