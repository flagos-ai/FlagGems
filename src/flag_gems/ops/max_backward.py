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
import math
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# Tiling used by the gradient of ``torch.max(input, dim)``: one program owns a
# ``_TILE_COLUMNS_MAX``-wide column tile of the contiguously written output slab
# and up to ``_TILE_LANES // _TILE_COLUMNS_MAX`` rows of the reduced axis. The
# split of the reduced axis is capped so the grid stays near
# ``_BACKWARD_TARGET_PROGRAMS`` programs, which is enough to fill the device
# without turning large cases into one-program-per-element launches.
_TILE_COLUMNS_MAX = 1024
_TILE_LANES = 1024
_BACKWARD_TARGET_PROGRAMS = 528

# The two-launch full-reduction path reduces all per-block counts inside one
# tile in the second kernel; beyond this many blocks it switches to the atomic
# count counter so the per-program partial load never becomes quadratic.
_MAX_PARTIALS = 4096


@libentry()
@triton.jit
def max_backward_lastdim_kernel(
    grad_ptr,
    index_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """Write the whole gradient slab for a reduction over the innermost dim.

    Program ``(o, pid)`` owns the ``BLOCK``-long slice ``[pid*BLOCK, pid*BLOCK +
    BLOCK)`` of row ``o`` and writes ``grad[o]`` at ``indices[o]`` and zero
    elsewhere, so the output is produced in a single pass: no zero-fill launch
    followed by a scatter. The block index is the second grid axis because the
    row count easily exceeds the 65535 limit of a CUDA y dimension.
    """
    row = ext.program_id(0)
    offsets = ext.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    grad = tl.load(grad_ptr + row)
    index = tl.load(index_ptr + row)
    value = tl.where(offsets.to(tl.int64) == index, grad, tl.zeros_like(grad))
    tl.store(out_ptr + row.to(tl.int64) * N + offsets, value, mask=mask)


@libentry()
@triton.jit
def max_backward_tile_kernel(
    grad_ptr,
    index_ptr,
    out_ptr,
    reduction,
    inner,
    chunk,
    BJ: tl.constexpr,
    BR: tl.constexpr,
):
    """Write the whole gradient slab when the reduced dim is interior.

    One program owns ``BJ`` columns of the ``outer`` plane and the ``chunk``-long
    slice of the reduced axis starting at ``pid_r * chunk``. The int64 argmax
    index and the upstream gradient are loaded once and reused across the loop,
    so index traffic stays at ``outer * inner`` elements instead of being re-read
    for every reduction step. The plane index is the first grid axis so it is not
    limited to the 65535 entries of a CUDA y dimension.
    """
    outer = ext.program_id(0)
    pid_j = ext.program_id(1)
    pid_r = ext.program_id(2)
    column = pid_j * BJ + tl.arange(0, BJ)
    column_mask = column < inner
    flat = outer.to(tl.int64) * inner + column
    index = tl.load(index_ptr + flat, mask=column_mask, other=-1)
    grad = tl.load(grad_ptr + flat, mask=column_mask, other=0.0)
    zero = tl.zeros_like(grad)
    base = out_ptr + outer.to(tl.int64) * (reduction * inner) + column
    start = pid_r * chunk
    for offset_r in range(start, start + chunk, BR):
        rows = offset_r + tl.arange(0, BR)
        row_mask = rows < reduction
        hit = (rows.to(tl.int64)[:, None] == index[None, :]) & row_mask[:, None]
        value = tl.where(hit, grad[None, :], zero[None, :])
        tl.store(
            base + rows[:, None] * inner,
            value,
            mask=row_mask[:, None] & column_mask[None, :],
        )


@libentry()
@triton.jit
def evenly_distribute_count_kernel(
    inp_ptr,
    result_ptr,
    count_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    result = tl.load(result_ptr)
    nan_result = result != result
    values = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    hits = ((values == result) | ((values != values) & nan_result)) & mask
    tl.atomic_add(count_ptr, tl.sum(hits.to(tl.int32), axis=0))


@libentry()
@triton.jit
def evenly_distribute_fill_kernel(
    inp_ptr,
    grad_ptr,
    result_ptr,
    count_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    result = tl.load(result_ptr)
    nan_result = result != result
    values = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    match = ((values == result) | ((values != values) & nan_result)) & mask
    scaled = _even_share(tl.load(grad_ptr), tl.load(count_ptr), out_ptr)
    tl.store(
        out_ptr + offsets,
        tl.where(match, scaled, tl.zeros_like(scaled)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _even_share(grad, count, out_ptr):
    """``grad_output / count`` computed in a wide enough accumulator.

    The quotient of an even split is evaluated in float32 for the 16-bit and
    32-bit dtypes and in float64 for a float64 input, so the result never loses
    precision relative to the element type it is stored as.
    """
    acc_type = tl.float64 if out_ptr.dtype.element_ty is tl.float64 else tl.float32
    return grad.to(acc_type) / count.to(acc_type)


@libentry()
@triton.jit
def evenly_distribute_slab_kernel(
    inp_ptr,
    grad_ptr,
    result_ptr,
    partial_ptr,
    out_ptr,
    n_elements,
    n_partials,
    BLOCK: tl.constexpr,
):
    """First pass that both counts the maxima and writes the gradient slab.

    Each program stores its own match count and the offset of its first match
    (plain ``tl.store``, no atomics and no zero-initialized counter), and writes
    the *unscaled* upstream gradient at the matching positions. The second pass
    therefore never has to read ``inp`` again for the (overwhelmingly common)
    single-match block.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    result = tl.load(result_ptr)
    nan_result = result != result
    values = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    hits = ((values == result) | ((values != values) & nan_result)) & mask
    tl.store(partial_ptr + pid, tl.sum(hits.to(tl.int32), axis=0))
    tl.store(
        partial_ptr + n_partials + pid,
        tl.min(tl.where(hits, offsets, n_elements), axis=0),
    )
    grad = tl.load(grad_ptr)
    tl.store(
        out_ptr + offsets,
        tl.where(hits, grad, tl.zeros_like(grad)),
        mask=mask,
    )


@libentry()
@triton.jit
def evenly_distribute_scale_kernel(
    inp_ptr,
    grad_ptr,
    result_ptr,
    partial_ptr,
    out_ptr,
    n_elements,
    n_partials,
    BLOCK: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    """Second pass: replace the matches with ``grad_output / count``.

    A block that holds exactly one match is fixed by a single scalar store using
    the offset recorded by the first pass. A block with several matches is
    rescanned, so every tie pattern stays correct.
    """
    pid = ext.program_id(0)
    partials = tl.arange(0, BLOCK_PARTIALS)
    partial_mask = partials < n_partials
    count = tl.sum(tl.load(partial_ptr + partials, mask=partial_mask, other=0), axis=0)
    scaled = _even_share(tl.load(grad_ptr), count, out_ptr)
    local = tl.load(partial_ptr + pid)
    if local == 1:
        target = tl.load(partial_ptr + n_partials + pid)
        tl.store(out_ptr + target, scaled.to(out_ptr.dtype.element_ty))
    elif local > 1:
        result = tl.load(result_ptr)
        nan_result = result != result
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        match = ((values == result) | ((values != values) & nan_result)) & mask
        tl.store(
            out_ptr + offsets,
            tl.where(match, scaled, tl.zeros_like(scaled)).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _device_context(device):
    """Guard for ``device``, skipped when it is already the active device.

    The autograd engine has selected the input's device before it calls
    ``backward``, and launching on the active device needs no guard. Entering one
    unconditionally costs more than the kernel launch itself for small inputs, so
    it is only taken when the tensor really lives on another device.
    """
    index = device.index
    lookup = getattr(torch_device_fn, "current_device", None)
    if index is None or lookup is None:
        return nullcontext()
    return nullcontext() if lookup() == index else torch_device_fn.device(device)


def _bit_length(value):
    """``int.bit_length`` for a non-negative value, without the builtin name."""
    return value.bit_length() if value > 0 else 0


def _backward_geometry(inp, dim):
    """Return ``(outer, reduction, inner)`` of ``dim`` in ``inp``'s layout."""
    ndim = inp.ndim
    dim = dim + ndim if dim < 0 else dim
    assert 0 <= dim < ndim, f"Invalid dim {dim} for a {ndim}-D tensor"
    shape = inp.shape
    outer = math.prod(shape[:dim])
    reduction = shape[dim]
    inner = math.prod(shape[dim + 1 :])
    return outer, reduction, inner


def max_backward(grad_output, self, result, indices, dim):
    """Gradient of ``torch.max(self, dim, keepdim)`` with respect to ``self``.

    ``grad_output`` holds the upstream gradient of the values output (already
    squeezed when the forward used ``keepdim=True``) and ``indices`` holds the
    int64 argmax positions saved by the forward pass. The whole gradient slab is
    written by one kernel, which is exactly ATen's
    ``zeros_like(self).scatter_(dim, indices, grad_output)``: ties follow the
    single index the forward selected, negative ``dim`` is normalized, and
    ``result`` belongs to the ATen signature but the backward math never needs
    it. Non-contiguous ``self`` is fine because only its geometry is used and the
    result is freshly allocated in the standard layout.
    """
    logger.debug("GEMS MAX_BACKWARD")
    outer, reduction, inner = _backward_geometry(self, dim)
    dim = dim + self.ndim if dim < 0 else dim
    grad = grad_output.contiguous()
    index = indices.contiguous()
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    if out.numel() == 0:
        return out
    with _device_context(self.device):
        if inner == 1:
            rows = outer * inner
            block = triton.next_power_of_2(reduction)
            if block > 1024:
                block = 1024
            grid = (rows, triton.cdiv(reduction, block))
            max_backward_lastdim_kernel[grid](grad, index, out, reduction, block)
        else:
            columns = 1 << _bit_length(inner - 1)
            if columns > _TILE_COLUMNS_MAX:
                columns = _TILE_COLUMNS_MAX
            lanes = _TILE_LANES // columns
            lanes = lanes if lanes > 1 else 1
            tiles = triton.cdiv(inner, columns)
            budget = _BACKWARD_TARGET_PROGRAMS // (tiles * outer)
            budget = budget if budget > 1 else 1
            splits = triton.cdiv(reduction, lanes)
            splits = splits if splits < budget else budget
            chunk = triton.cdiv(reduction, splits)
            splits = triton.cdiv(reduction, chunk)
            max_backward_tile_kernel[(outer, tiles, splits)](
                grad,
                index,
                out,
                reduction,
                inner,
                chunk,
                columns,
                lanes,
            )
    return out


def evenly_distribute_backward(grad_output, self, result):
    """Gradient of a full reduction such as ``torch.max(self)``.

    Every position holding the maximum receives an equal share of the upstream
    gradient; tie positions and (when the maximum is NaN) every NaN are matched
    with ATen's ``(self == result) | (self.isnan() & result.isnan())`` predicate,
    so the count is exactly the number of matched elements.

    The default two-launch path writes the gradient slab while it counts, so
    ``self`` is read once. For an input so large that the per-block partial
    counts would not fit the second kernel's single-tile reduction, the same
    formula runs as a zero-initialized atomic count followed by a plain fill.
    """
    logger.debug("GEMS EVENLY_DISTRIBUTE_BACKWARD")
    inp = self.contiguous()
    n_elements = inp.numel()
    out = torch.empty_like(inp)
    if n_elements == 0:
        return out
    grad = grad_output.contiguous().reshape(())
    value = result.contiguous().reshape(())
    block = 2048
    n_partials = triton.cdiv(n_elements, block)
    with _device_context(inp.device):
        if n_partials <= _MAX_PARTIALS:
            block_partials = triton.next_power_of_2(n_partials)
            partial = torch.empty(2 * n_partials, dtype=torch.int32, device=inp.device)
            grid = (n_partials,)
            evenly_distribute_slab_kernel[grid](
                inp, grad, value, partial, out, n_elements, n_partials, block
            )
            evenly_distribute_scale_kernel[grid](
                inp,
                grad,
                value,
                partial,
                out,
                n_elements,
                n_partials,
                block,
                block_partials,
            )
        else:
            count = torch.zeros([], dtype=torch.int32, device=inp.device)
            grid = (n_partials,)
            evenly_distribute_count_kernel[grid](inp, value, count, n_elements, block)
            evenly_distribute_fill_kernel[grid](
                inp, grad, value, count, out, n_elements, block
            )
    return out
