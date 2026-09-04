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

"""Kunlunxin(XPU) specialization of ``aten::roll``.

The generic implementation gathers every output element with a runtime
``offset // stride % size`` decode.  On XPU that lands on the discrete
gather path (~6-7 GB/s measured), so a 655M element roll costs ~368 ms
against ~2.5 ms for the vendor fused kernel (speedup 0.006x).

``roll`` is a pure permutation, so it is expressed here as *whole buffer
rotation* plus a small wrap fix-up:

* ``out_flat[i] = in_flat[(i - delta) mod numel]`` with
  ``delta = sum(shift[d] * stride[d])`` is already the exact answer for
  every element whose index along each rolled dim ``d != 0`` did not wrap
  (the wrap of dim 0 is absorbed by ``mod numel``).  That is two fully
  contiguous block copies.
* The remaining ``2**k - 2`` blocks (some non-zero dim wrapped) are tiny
  strided copies.

Both use ``torch.ops.aten._copy_from``, which flag_gems never overrides, so
they go straight to the vendor strided-copy engine (~1.85 TB/s measured).

One XPU specific trap: that engine only reaches full bandwidth when the
*destination* byte offset is 32-byte aligned.  A rotation destination is
offset by ``delta`` elements, and an unaligned destination costs a hard 2.3x
(268M fp16: 0.58 ms at dst offset 0/16/32/... vs 1.33 ms at dst offset
1/2/4/8).  The output buffer is therefore over-allocated by up to
``32 / itemsize - 1`` elements so that the bulk piece writes to an aligned
address; the returned tensor is a contiguous view of that buffer.

For payloads that fit in a few tiles the fixed per-copy launch cost of four
``_copy_from`` calls dominates, so small tensors keep a single fused Triton
gather kernel instead.
"""

import itertools
import logging
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

IntOrInts = int | Sequence[int]

# aten::_copy_from on XPU only reaches peak bandwidth with a 32B aligned dst.
_DST_ALIGN_BYTES = 32
# Below this payload the four _copy_from launches cost more than one gather.
_TRITON_MAX_BYTES = 1 << 16
# Padding the output only pays off once the copy is bandwidth bound.
_ALIGN_MIN_BYTES = 1 << 18
# Number of wrap dims the fused gather kernel understands.
_MAX_WRAP_DIMS = 4
_TRITON_BLOCK = 512


def roll(inp: torch.Tensor, shifts, dims=None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ROLL")

    _validate_inputs(inp, shifts, dims)
    shape = tuple(inp.shape)
    numel = inp.numel()
    src = _contiguous(inp)

    if dims is None or _is_empty_sequence(dims):
        flat = src.reshape(-1)
        n = flat.numel()
        shift = 0 if n == 0 else _as_tuple(shifts)[0] % n
        return _rotate_flat(flat, shift).view(shape)

    effective = _effective_shifts(shape, _as_tuple(shifts), _as_tuple(dims), numel == 0)
    active = [
        (dim, shift) for dim, shift in enumerate(effective) if shift and shape[dim]
    ]

    if numel == 0 or not active:
        out = torch.empty_like(src)
        if numel:
            torch.ops.aten._copy_from(src, out, False)
        return out

    strides = src.stride()
    delta = sum(effective[dim] * strides[dim] for dim, _ in active) % numel
    wrap_dims = [dim for dim, _ in active if dim != 0]

    if not wrap_dims:
        # dim-0 rolls are exactly a flat rotation, no fix-up needed.
        return _rotate_flat(src.reshape(-1), delta).view(shape)

    if (
        numel * src.element_size() <= _TRITON_MAX_BYTES
        and len(wrap_dims) <= _MAX_WRAP_DIMS
    ):
        return _roll_gather(src, numel, delta, effective, wrap_dims)

    out = _rotate_flat(src.reshape(-1), delta).view(shape)
    for combo in _fixup_blocks(shape, active):
        src_view = src
        out_view = out
        for dim, out_start, in_start, length in combo:
            src_view = src_view.narrow(dim, in_start, length)
            out_view = out_view.narrow(dim, out_start, length)
        torch.ops.aten._copy_from(src_view, out_view, False)
    return out


def _rotate_flat(flat: torch.Tensor, delta: int) -> torch.Tensor:
    """``out[i] = flat[(i - delta) % n]`` with two contiguous block copies."""
    n = flat.numel()
    out = _empty_rotated(flat, n, delta)
    if n == 0:
        return out
    if delta == 0:
        torch.ops.aten._copy_from(flat, out, False)
        return out
    torch.ops.aten._copy_from(flat[n - delta :], out[:delta], False)
    torch.ops.aten._copy_from(flat[: n - delta], out[delta:], False)
    return out


def _empty_rotated(ref: torch.Tensor, numel: int, delta: int) -> torch.Tensor:
    """Allocate ``numel`` elements so the bulk rotation piece lands aligned."""
    pad = 0
    itemsize = ref.element_size()
    if (
        delta
        and numel - delta > delta
        and numel * itemsize >= _ALIGN_MIN_BYTES
        and _DST_ALIGN_BYTES > itemsize
    ):
        unit = _DST_ALIGN_BYTES // itemsize
        pad = -delta % unit
    if pad == 0:
        return torch.empty(numel, dtype=ref.dtype, device=ref.device)
    buf = torch.empty(numel + pad, dtype=ref.dtype, device=ref.device)
    return buf.narrow(0, pad, numel)


def _fixup_blocks(shape: Sequence[int], active: Sequence[tuple]) -> list:
    """Blocks the flat rotation does not already produce.

    Each entry is one ``(dim, out_start, in_start, length)`` tuple per active
    dim.  The rotation covers every combination that does not wrap on a dim
    other than dim 0, so only the rest is returned.
    """
    per_dim = []
    for dim, shift in active:
        size = shape[dim]
        no_wrap = (dim, shift, 0, size - shift)
        wrapped = (dim, 0, size - shift, shift)
        per_dim.append((no_wrap, wrapped))

    blocks = []
    for combo in itertools.product(*per_dim):
        wrapped_outer = any(
            dim != 0 and out_start == 0 for dim, out_start, _, _ in combo
        )
        if wrapped_outer:
            blocks.append(combo)
    return blocks


def _roll_gather(
    src: torch.Tensor,
    numel: int,
    delta: int,
    effective: Sequence[int],
    wrap_dims: Sequence[int],
) -> torch.Tensor:
    out = torch.empty_like(src)
    strides = src.stride()
    params = []
    for index in range(_MAX_WRAP_DIMS):
        if index < len(wrap_dims):
            dim = wrap_dims[index]
            params.extend((src.size(dim), strides[dim], effective[dim]))
        else:
            params.extend((1, 1, 0))
    need_mask = numel % _TRITON_BLOCK != 0
    grid = (triton.cdiv(numel, _TRITON_BLOCK),)
    _roll_gather_kernel[grid](
        src.reshape(-1),
        out.reshape(-1),
        numel,
        delta,
        *params,
        NWRAP=len(wrap_dims),
        BLOCK=_TRITON_BLOCK,
        NEED_MASK=need_mask,
    )
    return out


@libentry()
@triton.jit
def _roll_gather_kernel(
    in_ptr,
    out_ptr,
    numel,
    delta,
    size0,
    stride0,
    shift0,
    size1,
    stride1,
    shift1,
    size2,
    stride2,
    shift2,
    size3,
    stride3,
    shift3,
    NWRAP: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = offsets - delta
    if NWRAP >= 1:
        source += tl.where((offsets // stride0) % size0 < shift0, size0 * stride0, 0)
    if NWRAP >= 2:
        source += tl.where((offsets // stride1) % size1 < shift1, size1 * stride1, 0)
    if NWRAP >= 3:
        source += tl.where((offsets // stride2) % size2 < shift2, size2 * stride2, 0)
    if NWRAP >= 4:
        source += tl.where((offsets // stride3) % size3 < shift3, size3 * stride3, 0)
    source = tl.where(source < 0, source + numel, source)
    if NEED_MASK:
        # Clamp instead of relying on masked loads: XPU ignores `other=` on
        # some paths, and the store mask already discards the tail lanes.
        source = tl.minimum(tl.maximum(source, 0), numel - 1)
        tl.store(out_ptr + offsets, tl.load(in_ptr + source), mask=offsets < numel)
    else:
        tl.store(out_ptr + offsets, tl.load(in_ptr + source))


def _contiguous(inp: torch.Tensor) -> torch.Tensor:
    if inp.is_contiguous():
        return inp
    out = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if inp.numel():
        torch.ops.aten._copy_from(inp, out, False)
    return out


def _effective_shifts(
    shape: Sequence[int],
    shifts: Sequence[int],
    dims: Sequence[int],
    allow_empty_wrap: bool,
) -> list[int]:
    ndim = len(shape)
    effective = [0] * ndim
    for shift, dim in zip(shifts, dims):
        effective[_canonicalize_dim(dim, ndim, allow_empty_wrap)] += shift
    for index, size in enumerate(shape):
        if size:
            effective[index] %= size
    return effective


def _canonicalize_dim(dim: int, ndim: int, allow_empty_wrap: bool = False) -> int:
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if allow_empty_wrap:
        return dim % ndim
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _validate_inputs(inp: torch.Tensor, shifts, dims=None) -> None:
    if not isinstance(inp, torch.Tensor):
        raise TypeError("roll(): argument 'input' must be Tensor")
    if not _is_int_or_int_sequence(shifts):
        raise TypeError("roll(): argument 'shifts' must be int or tuple of ints")
    shift_count = 1 if isinstance(shifts, int) else len(shifts)
    if shift_count == 0:
        raise RuntimeError("`shifts` required")

    if dims is None or _is_empty_sequence(dims):
        if shift_count > 1:
            raise RuntimeError(
                f"shifts and dimensions must align. shifts: {shift_count}, dims:0"
            )
        return

    if not _is_int_or_int_sequence(dims):
        raise TypeError("roll(): argument 'dims' must be int or tuple of ints")
    dim_count = 1 if isinstance(dims, int) else len(dims)
    if shift_count != dim_count:
        raise RuntimeError("shifts and dimensions must align")


def _as_tuple(value: IntOrInts) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _is_int_or_int_sequence(value: object) -> bool:
    if isinstance(value, int):
        return True
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, int) for item in value)


def _is_empty_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence) and not isinstance(value, int) and len(value) == 0
    )
