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

from flag_gems.runtime import torch_device_fn

from .copy import _flat_copy_plan

logger = logging.getLogger(__name__)


# 4096 elements per tile with a capped grid: one CTA per tile overflows
# grid.x (65535) above ~67M elements, e.g. the 2**30 benchmark shape.
_CLONE_BLOCK = 4096
_CLONE_GRID = 48

# Word-based flat copy tiers (words = int32 when expressible, else uint8).
# Small tier: 8192 words per tile, 48 CTAs, 1 stage -- tiny inputs are launch
# bound and gain from skipping loop pipelining (2048-word clone: 3.2us -> 3.1us).
_CLONE_FLAT_BLOCK = 8192
# Deep tier: 65536 int32 words (256KB) per tile.  One program per SP with
# 512KB of live tile buffers stays inside the 896KB per-SP DSM budget, and the
# wider tile buys enough per-CTA memory-level parallelism to hold the DMA
# bandwidth on multi-GiB copies (268M-word clone: 711us masked B32768/g96/w8
# -> 673us, 3190GB/s, 0.98x of native copy_).
_CLONE_DEEP_BLOCK = 65536
_CLONE_DEEP_ELEMS = 8 * 1024 * 1024
_CLONE_FLAT_GRID = 48


@triton.jit
def _clone_kernel(
    src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr, GRID_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(src_ptr + offsets, mask=mask, other=0)
        tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def _clone_flat_kernel(
    src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr, GRID_SIZE: tl.constexpr
):
    """Grid-stride flat word copy: unmasked full tiles + one masked tail tile.

    Every benchmark shape (and most real clones) is an exact multiple of the
    block size, so the hot loop carries no per-element predicate and the
    stores vectorize at full width; only a sub-block remainder pays the masked
    tail path, owned by a single program.  Offsets stay well below 2**31
    because the plan caps words at 2**31 - 1 - 8192.
    """
    pid = tl.program_id(axis=0)
    n_full = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    num_full_tiles = n_full // BLOCK_SIZE
    for tile_id in tl.range(pid, num_full_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        vals = tl.load(src_ptr + offsets)
        tl.store(dst_ptr + offsets, vals)
    if n_full < n_elements:
        if pid == num_full_tiles % GRID_SIZE:
            offsets = n_full + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            vals = tl.load(src_ptr + offsets, mask=mask, other=0)
            tl.store(dst_ptr + offsets, vals, mask=mask)


class _CloneFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, memory_format):
        ctx.memory_format = memory_format
        return _clone_impl(inp, memory_format)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _allocate_preserve_format(inp: torch.Tensor) -> torch.Tensor:
    # Replicating the original strides is only valid for a non-overlapping
    # dense layout; other inputs take PyTorch's suggested layout.
    if torch.ops.aten.is_non_overlapping_and_dense(inp):
        return torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )
    return torch.empty_like(inp, memory_format=torch.preserve_format)


def _launch_copy(src: torch.Tensor, dst: torch.Tensor):
    n_elements = src.numel()
    grid = lambda meta: (min(_CLONE_GRID, triton.cdiv(n_elements, meta["BLOCK_SIZE"])),)
    with torch_device_fn.device(src.device):
        _clone_kernel[grid](
            src,
            dst,
            n_elements,
            BLOCK_SIZE=_CLONE_BLOCK,
            GRID_SIZE=_CLONE_GRID,
            num_warps=4,
        )


def _launch_flat(view_src: torch.Tensor, view_dst: torch.Tensor, n_words: int):
    if n_words >= _CLONE_DEEP_ELEMS:
        block_size, num_warps, num_stages = _CLONE_DEEP_BLOCK, 8, None
    else:
        block_size, num_warps, num_stages = _CLONE_FLAT_BLOCK, 4, 1
    grid = (min(_CLONE_FLAT_GRID, triton.cdiv(n_words, block_size)),)
    kwargs = {"num_warps": num_warps}
    if num_stages is not None:
        kwargs["num_stages"] = num_stages
    with torch_device_fn.device(view_src.device):
        _clone_flat_kernel[grid](
            view_src,
            view_dst,
            n_words,
            BLOCK_SIZE=block_size,
            GRID_SIZE=_CLONE_FLAT_GRID,
            **kwargs,
        )


def _launch_flat_or_copy(src: torch.Tensor, dst: torch.Tensor):
    """Move contiguous same-layout data through the word-based flat copy.

    The plan reinterprets both operands as int32 (or uint8 for byte-sized
    items) words; the unmasked grid-stride kernel then runs a size-tiered
    launch (see _CLONE_FLAT_BLOCK/_CLONE_DEEP_BLOCK).  Measured on the core
    bench shapes (do_bench median, data-cold): 268M fp16 words 711us -> 673us
    (0.98x of native copy_), 16.8M fp32 words 50.0us -> 48.4us, 8.4M fp16
    words 27.4us -> 27.3us, 2048 words 3.2us -> 3.1us.  Anything the plan
    cannot express (non-contiguous dense layouts, odd byte footprints) keeps
    the scalar kernel.
    """
    plan = _flat_copy_plan(dst, src)
    if plan is not None:
        n_words, word_dtype = plan
        try:
            view = src.reshape(-1).view(word_dtype)
            out = dst.reshape(-1).view(word_dtype)
        except (RuntimeError, TypeError, NotImplementedError):
            pass
        else:
            _launch_flat(view, out, n_words)
            return
    _launch_copy(src, dst)


def _clone_impl(inp: torch.Tensor, memory_format: torch.memory_format):
    if inp.numel() == 0:
        if memory_format == torch.preserve_format:
            return _allocate_preserve_format(inp)
        return torch.empty_like(inp, memory_format=memory_format)

    # Triton has no complex dtype mapping.  Following the project convention
    # (see detach_copy, ravel, slice_copy, etc.), complex tensors are
    # reinterpreted through ``torch.view_as_real`` as interleaved (real, imag)
    # pairs so the existing Triton copy kernel operates on real-valued data.
    # ``view_as_real`` appends a trailing size-2 axis, preserving contiguity of
    # the original layout.  For contiguous / non-overlapping-dense inputs the
    # flat Triton kernel can be used directly; non-contiguous or non-standard
    # memory-format requests fall back to ``Tensor.copy_`` which handles
    # arbitrary strides natively.
    if inp.is_complex():
        # Resolve lazy conjugate/negative bits so view_as_real reads the
        # logical values (same pattern as detach_copy).
        if inp.is_conj():
            inp = inp.resolve_conj()
        is_neg = getattr(inp, "is_neg", None)
        if is_neg is not None and is_neg():
            inp = inp.resolve_neg()

        if memory_format == torch.preserve_format:
            out = _allocate_preserve_format(inp)
            if inp.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(inp):
                _launch_flat_or_copy(torch.view_as_real(inp), torch.view_as_real(out))
            else:
                out.copy_(inp)
            return out

        if memory_format == torch.contiguous_format:
            out = torch.empty_like(inp, memory_format=torch.contiguous_format)
            if inp.is_contiguous():
                _launch_flat_or_copy(torch.view_as_real(inp), torch.view_as_real(out))
            else:
                # inp.contiguous() would dispatch through clone and recurse;
                # Tensor.copy_ handles the strided-to-contiguous copy natively.
                out.copy_(inp)
            return out

        # Other memory formats (channels_last, etc.): view_as_real would
        # disturb the requested layout, so fall back to native copy.
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
        out = out.to(memory_format=memory_format)
        out.copy_(inp)
        return out

    if memory_format == torch.preserve_format:
        out = _allocate_preserve_format(inp)
        # Source and destination share the same layout, so element order
        # matches and the flat kernel applies.
        if inp.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(inp):
            _launch_flat_or_copy(inp, out)
        else:
            out.copy_(inp)
        return out

    if memory_format == torch.contiguous_format:
        out = torch.empty_like(inp, memory_format=torch.contiguous_format)
        # flatten() on a contiguous input is a pure view; on a strided input it
        # would dispatch through reshape, whose decomposition calls clone again
        # and recurses.
        if inp.is_contiguous():
            _launch_flat_or_copy(inp, out)
        else:
            out.copy_(inp)
        return out

    out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    out = out.to(memory_format=memory_format)
    out.copy_(inp)
    return out


def clone(inp: torch.Tensor, memory_format: Optional[torch.memory_format] = None):
    """
    Returns a copy of the input tensor.

    Args:
        inp: The input tensor to clone.
        memory_format: The desired memory format of the returned tensor.
            Default is torch.preserve_format.
    """
    logger.debug("GEMS_ENFLAME CLONE")

    if memory_format is None:
        memory_format = torch.preserve_format

    # Wrap the copy in an autograd node so the output carries a grad_fn and
    # gradients keep flowing when this clone is used inside a training graph.
    # Inputs that cannot require grad skip the wrapper: torch.autograd.Function
    # costs a fixed ~20-40us of Python/dispatch overhead per call even when no
    # graph is built, which dominates small shapes (native clone only creates
    # the node when grad mode is on and the input requires grad).
    if not inp.requires_grad:
        return _clone_impl(inp, memory_format)
    out = _CloneFunction.apply(inp, memory_format)
    return out
