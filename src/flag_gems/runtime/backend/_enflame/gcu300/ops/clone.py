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

"""gcu300 ``clone`` with a capped grid.

The generic ``flag_gems.ops.clone`` launches one CTA per 1024-element tile, so
a flat copy larger than 65535 * 1024 elements (67M) asks the GCU launcher for a
grid.x above the 65535 hardware limit and fails with

    triton.runtime.errors.OutOfResources: out of resource: grid.x,
    Required: 73728, Hardware limit: 65535

That is hit by the Qwen3.6-35B-A3B vision-encoder warmup on S60 (gcu300): the
patch-embedding clone of a 75.5M-element tensor needs 73,728 CTAs. gcu400 has
its own capped clone (``_enflame/gcu400/ops/clone.py``); gcu300 previously fell
through to the generic implementation.

Like the other backend ops, this module carries a self-contained implementation
and is exported from ``ops/__init__.py``, which overrides the generic op through
the normal backend registration path.  A fixed number of CTAs
(``_CLONE_GRID``) each walk every ``_CLONE_GRID``-th tile, so the launch shape
is independent of the tensor size.  Small copies keep a single-launch shape as
well because the tile count is clamped by the grid size.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# 4096 elements per tile keeps the per-CTA DMA request large enough to stay
# bandwidth bound, while a 48-CTA grid stays far below the 65535 grid.x limit
# for any tensor size.
_CLONE_BLOCK = 4096
_CLONE_GRID = 48


@triton.jit
def _clone_kernel(
    src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr, GRID_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(src_ptr + offsets, mask=mask)
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


def _is_fake_or_meta(inp: torch.Tensor) -> bool:
    """Whether ``inp`` carries metadata only, with no storage to read or write.

    ``aten::clone`` is a CompositeExplicitAutograd op whose Meta implementation
    is a "default backend kernel": shape propagation re-runs the backend
    implementation under the fake/meta mode instead of using a dedicated meta
    function.  Because FlagGems takes over the device dispatch key, that
    fallback lands in this module, and the Triton launch below would ask a
    FakeTensor for ``data_ptr()`` -- which raises.  Handle this case by
    deriving the output layout only.

    ``FakeTensor`` is detected by its ``fake_mode`` attribute rather than by
    importing the class: it lives in a private module that has moved across
    torch releases, and a real meta tensor (``is_meta``) has no ``fake_mode``,
    so both spellings are covered without a version-sensitive import.  A plain
    eager tensor on a real device has neither attribute and never takes this
    path.
    """
    return inp.is_meta or hasattr(inp, "fake_mode")


def _clone_meta(inp: torch.Tensor, memory_format: torch.memory_format) -> torch.Tensor:
    """Allocate the output for a metadata-only input without launching kernels."""
    if memory_format == torch.preserve_format:
        return _allocate_preserve_format(inp)
    return torch.empty_like(inp, memory_format=memory_format)


def _launch_copy(src: torch.Tensor, dst: torch.Tensor):
    n_elements = src.numel()
    num_tiles = triton.cdiv(n_elements, _CLONE_BLOCK)
    grid = (min(_CLONE_GRID, max(1, num_tiles)),)
    with torch_device_fn.device(src.device):
        _clone_kernel[grid](
            src,
            dst,
            n_elements,
            BLOCK_SIZE=_CLONE_BLOCK,
            GRID_SIZE=_CLONE_GRID,
            num_warps=4,
        )


def _clone_impl(inp: torch.Tensor, memory_format: torch.memory_format):
    if inp.numel() == 0:
        if memory_format == torch.preserve_format:
            return _allocate_preserve_format(inp)
        return torch.empty_like(inp, memory_format=memory_format)

    # Metadata-only inputs (FakeTensor under torch.compile/export, or a tensor
    # on the meta device) have no storage: every branch below would either
    # launch a Triton kernel or dispatch a copy that eventually reads a data
    # pointer.  Resolve the layout here instead.
    if _is_fake_or_meta(inp):
        return _clone_meta(inp, memory_format)

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
                _launch_copy(torch.view_as_real(inp), torch.view_as_real(out))
            else:
                out.copy_(inp)
            return out

        if memory_format == torch.contiguous_format:
            out = torch.empty_like(inp, memory_format=torch.contiguous_format)
            if inp.is_contiguous():
                _launch_copy(torch.view_as_real(inp), torch.view_as_real(out))
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
            _launch_copy(inp, out)
        else:
            out.copy_(inp)
        return out

    if memory_format == torch.contiguous_format:
        out = torch.empty_like(inp, memory_format=torch.contiguous_format)
        # flatten() on a contiguous input is a pure view; on a strided input it
        # would dispatch through reshape, whose decomposition calls clone again
        # and recurses.
        if inp.is_contiguous():
            _launch_copy(inp, out)
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
    out = _CloneFunction.apply(inp, memory_format)
    return out
