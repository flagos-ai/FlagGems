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

"""Resize-and-copy using the Lazy backend's eager materialization recipe.

PyTorch has no native CPU/CUDA kernel for this private operator. The destination
is resized in place and returned; the backend-local ``copy_`` performs the data
movement and dtype conversion. Existing storage can be reused by ``resize_``.

Same-dtype contiguous copies (the only large case) take a dedicated flat kernel
here instead of the generic ``flag_gems.ops.copy.copy_``: that one launches one
tiny program per 16 KB tile (4096 programs for a 64 MB tensor), which serialises
into ~85 waves on gcu400's 48 SPs and leaves ~0.6 TB/s on the table. The tuned
kernel keeps the backend-local 48-program grid-stride form, widens the tile to
32 KB and drops the per-tile mask on full tiles (measured 215 -> ~53 us at
64 MB, native recipe 49 us). Everything else (dtype cast, strided or aliased
operands, empty tails) falls back to ``.copy``'s ``copy_`` unchanged.

The ``out`` variant mirrors the result into ``out`` as well, so the eager recipe
moves every byte twice (src -> dst, then dst -> out; 4N traffic in N words).
When both legs qualify for the flat path the two are fused into one kernel that
reads the source once and stores to both destinations (3N traffic, one launch
instead of two): measured 1.37-1.66x over the native recipe across the core
shapes, e.g. fp32 64 MB 93 -> 68 us and fp16 2 MB 9.3 -> 5.7 us.
"""

import logging
import warnings

import torch
import triton
import triton.language as tl

from .copy import _flat_copy_plan, copy_

logger = logging.getLogger(__name__)

# 8192 int32 (32 KB) per program-iteration; DSM peak ~64 KB (load+store tiles),
# far below the 448 KB budget that keeps the full 48-way program concurrency.
_RESIZE_COPY_BLOCK = 8192
_RESIZE_COPY_GRID = 48
# Offsets are int32, so the masked tail tile must still fit in an int32.
_RESIZE_COPY_MAX_ELEMS = 2**31 - 1 - _RESIZE_COPY_BLOCK


@triton.jit
def _resize_flat_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Grid-stride flat copy over int32/uint8 words with an unmasked main loop.

    Full tiles are copied without masks (masked accesses are slower on gcu400);
    only the single tail tile is masked. Plain offset loads/stores stay on the
    synchronous port, so unlike the block-pointer (DTE) transfer they cannot
    overrun the allocation and fault the MMU.
    """
    pid = tl.program_id(0)
    num_full = n_elements // BLOCK_SIZE
    for tile_id in tl.range(pid, num_full, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(src_ptr + offsets)
        tl.store(dst_ptr + offsets, value)
    rem = n_elements - num_full * BLOCK_SIZE
    if pid == 0 and rem > 0:
        offsets = num_full * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(src_ptr + offsets, mask=mask, other=0)
        tl.store(dst_ptr + offsets, value, mask=mask)


@triton.jit
def _resize_flat_copy2_kernel(
    src_ptr,
    dst_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Grid-stride flat fan-out copy: read the source once, store twice.

    The ``out`` variant of the operator needs the same bytes in ``dst`` and in
    ``out``; the eager recipe pays 4N traffic (two full copies) while this
    kernel moves 3N (one read, two stores) in a single launch. Same unmasked
    main loop + single masked tail tile as the single-destination kernel; the
    value tile is stored to both destinations from the same registers.
    """
    pid = tl.program_id(0)
    num_full = n_elements // BLOCK_SIZE
    for tile_id in tl.range(pid, num_full, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        value = tl.load(src_ptr + offsets)
        tl.store(dst_ptr + offsets, value)
        tl.store(out_ptr + offsets, value)
    rem = n_elements - num_full * BLOCK_SIZE
    if pid == 0 and rem > 0:
        offsets = num_full * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(src_ptr + offsets, mask=mask, other=0)
        tl.store(dst_ptr + offsets, value, mask=mask)
        tl.store(out_ptr + offsets, value, mask=mask)


def _fast_resize_copy(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Tuned flat copy for the post-resize fast path.

    Returns True when the copy was performed (or trivially skipped). Guards
    mirror ``.copy``'s ``_can_use_triton`` plus ``_flat_copy_plan``; anything
    excluded here is handled by ``copy_`` with its usual fallbacks.
    """
    if src.device != dst.device:
        return False
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        # Preserve PyTorch's complex-cast warning semantics via the redispatch
        # path inside copy_.
        return False
    if torch._C._is_alias_of(dst, src):
        # Overlapping writes need aten's well-defined semantics; copy_ handles
        # the exact-alias no-op as well.
        return False
    plan = _flat_copy_plan(dst, src)
    if plan is None:
        return False
    n_elements, word_dtype = plan
    if n_elements > _RESIZE_COPY_MAX_ELEMS:
        return False
    if n_elements == 0:
        # dst was resized to src's shape, so the broadcast is trivially valid
        # and there is nothing to move.
        return True
    try:
        view = src.reshape(-1).view(word_dtype)
        out = dst.reshape(-1).view(word_dtype)
    except (RuntimeError, TypeError, NotImplementedError):
        return False
    grid = (min(_RESIZE_COPY_GRID, triton.cdiv(n_elements, _RESIZE_COPY_BLOCK)),)
    _resize_flat_copy_kernel[grid](
        view,
        out,
        n_elements,
        BLOCK_SIZE=_RESIZE_COPY_BLOCK,
        GRID_SIZE=_RESIZE_COPY_GRID,
        num_warps=4,
    )
    return True


def _fast_resize_copy2(dst: torch.Tensor, out: torch.Tensor, src: torch.Tensor) -> bool:
    """Fused fan-out copy for the ``out`` variant: fill ``dst`` and ``out`` from
    ``src`` with one kernel (3N traffic instead of the recipe's 4N).

    Returns True when both destinations were filled (or trivially skipped).
    Guards mirror ``_fast_resize_copy`` plus the second destination; anything
    excluded here is handled by the two-step path (``_fast_resize_copy`` x2 or
    ``copy_`` with its usual fallbacks).
    """
    if src.device != dst.device or src.device != out.device:
        return False
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if out.layout != torch.strided:
        return False
    if dst.is_quantized or src.is_quantized or out.is_quantized:
        return False
    if src.is_complex() or dst.is_complex() or out.is_complex():
        # Preserve PyTorch's complex-cast warning semantics via the redispatch
        # path inside copy_.
        return False
    if (
        torch._C._is_alias_of(dst, src)
        or torch._C._is_alias_of(out, src)
        or torch._C._is_alias_of(dst, out)
    ):
        # Overlapping writes need aten's well-defined semantics; the two-step
        # path handles each leg with its own alias handling.
        return False
    plan = _flat_copy_plan(dst, src)
    if plan is None:
        return False
    plan_out = _flat_copy_plan(out, src)
    if plan_out is None or plan_out != plan:
        return False
    n_elements, word_dtype = plan
    if n_elements > _RESIZE_COPY_MAX_ELEMS:
        return False
    if n_elements == 0:
        # Both tensors were resized to src's shape, so the broadcast is
        # trivially valid and there is nothing to move.
        return True
    try:
        view = src.reshape(-1).view(word_dtype)
        dst_view = dst.reshape(-1).view(word_dtype)
        out_view = out.reshape(-1).view(word_dtype)
    except (RuntimeError, TypeError, NotImplementedError):
        return False
    grid = (min(_RESIZE_COPY_GRID, triton.cdiv(n_elements, _RESIZE_COPY_BLOCK)),)
    _resize_flat_copy2_kernel[grid](
        view,
        dst_view,
        out_view,
        n_elements,
        BLOCK_SIZE=_RESIZE_COPY_BLOCK,
        GRID_SIZE=_RESIZE_COPY_GRID,
        num_warps=4,
    )
    return True


def _copy_from_and_resize(self: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Copy ``self`` into ``dst``, resizing ``dst`` to ``self``'s shape first.

    Returns ``dst`` itself; the caller's tensor object is mutated in place
    (``self``'s shape, ``dst``'s dtype preserved).
    """
    logger.debug("GEMS_ENFLAME _COPY_FROM_AND_RESIZE")
    dst.resize_(self.shape)
    if not _fast_resize_copy(dst, self):
        copy_(dst, self)
    return dst


def _copy_from_and_resize_out(
    self: torch.Tensor, dst: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    """``out`` variant: resize-and-copy ``self`` into ``dst``, then mirror the
    result into ``out`` (resized in place) and return ``out``.

    When both legs qualify for the flat fast path, ``dst`` and ``out`` are
    filled together by one fan-out kernel (each source byte read once from
    DRAM, stored to both destinations); otherwise the eager recipe's two-step
    copy runs unchanged. Option validation order and observable state at a
    raise match torchgen's generated out wrapper: warn, resize ``out``, then
    validate, with ``dst`` already materialized before any error surfaces.
    """
    logger.debug("GEMS_ENFLAME _COPY_FROM_AND_RESIZE_OUT")
    dst.resize_(self.shape)
    if out is dst:
        if not _fast_resize_copy(dst, self):
            copy_(dst, self)
        return out
    # Match torchgen's generated out wrapper: resize before validating options.
    if out.shape != dst.shape and out.numel() != 0:
        warnings.warn(
            "An output with one or more elements was resized since its shape "
            "does not match the required output shape. This behavior is deprecated.",
            UserWarning,
            stacklevel=2,
        )
    out.resize_(dst.shape)
    if out.dtype != dst.dtype or out.device != dst.device:
        # torchgen validates out's options after the functional call, so dst
        # must already hold the result before the error surfaces.
        if not _fast_resize_copy(dst, self):
            copy_(dst, self)
        if out.dtype != dst.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {dst.dtype}, "
                f"but got {out.dtype} instead"
            )
        raise RuntimeError(
            f"Expected out tensor to have device {dst.device}, "
            f"but got {out.device} instead"
        )
    if _fast_resize_copy2(dst, out, self):
        return out
    if not _fast_resize_copy(dst, self):
        copy_(dst, self)
    if not _fast_resize_copy(out, dst):
        copy_(out, dst)
    return out
