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

"""Kunlunxin (XPU) specialization of ``aten::flip``.

Two kernels, picked by ``_plan``:

``_flip_affine_kernel``
    Used whenever the flipped dimensions form a prefix or a suffix of a
    contiguous input -- which covers the whole-tensor reversal that dominates
    real shapes.  Such a flip is an affine map of the output index onto the
    input: a *row width* ``R``, a *row stride* ``in_step`` and a base offset,
    with the row optionally reversed.  That needs no per-dimension loop and no
    metadata, so the address math is a divide, a multiply and an add.

``_flip_meta_kernel``
    General fallback for a flip that is neither a prefix nor a suffix of the
    dims (e.g. ``dims=(0, 2)``), or for a non-contiguous input.  It walks the
    dimension metadata on the device.

The metadata is built on the CPU and only then moved to the device.  Building
it directly with ``torch.tensor(..., device="cuda")`` costs ~190 ms per call in
this backend: the tensor construction dispatches through ``lift_fresh`` while
GEMS is active, which lands a JIT compile inside the timed region.  Every shape
of the core benchmark paid that fixed cost, which is why a 64x64 flip and a
16 M element flip both measured ~200 ms.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Tile width and warp count measured on a P800 over both ends of the size
# range.  A 1 M element row-reversal gather needs >= 1024 lanes per program to
# amortise the launch; below that the grid is what saturates instead, and 256
# lanes with two warps is fastest for the small shapes (0.020 ms for 4096
# elements vs 0.031 ms for the vendor kernel).
_LARGE_BLOCK = 1024
_SMALL_BLOCK = 256
_SMALL_LIMIT = 1 << 16
_INT32_MAX = 2**31 - 1

# Marker returned by ``_plan`` when the affine mapping does not apply.
_GENERIC = "generic"


@libentry()
@triton.jit
def _flip_affine_kernel(
    inp,
    out,
    n_elements,
    in_base,
    in_step,
    R: tl.constexpr,
    REVERSED: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_I64: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    if USE_I64:
        linear = offsets.to(tl.int64)
    else:
        linear = offsets
    if R == 1:
        if REVERSED:
            src = in_base - linear * in_step
        else:
            src = in_base + linear * in_step
    else:
        row = linear // R
        col = linear - row * R
        if REVERSED:
            col = (R - 1) - col
        src = in_base + row * in_step + col
    value = tl.load(inp + src, mask=mask)
    tl.store(out + offsets, value, mask=mask)


@libentry()
@triton.jit
def _flip_meta_kernel(
    inp,
    out,
    meta,
    n_elements,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    linear = offsets
    inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)

    for dim in tl.static_range(NDIM - 1, -1, -1):
        size = tl.load(meta + dim)
        stride = tl.load(meta + NDIM + dim)
        should_flip = tl.load(meta + 2 * NDIM + dim)
        coord = linear % size
        linear //= size
        coord = tl.where(should_flip != 0, size - 1 - coord, coord)
        inp_offsets += coord * stride

    value = tl.load(inp + inp_offsets, mask=mask)
    tl.store(out + offsets, value, mask=mask)


def _plan(A: torch.Tensor, flip_dims):
    """Map a contiguous flip onto ``(in_base, in_step, R, reversed)``.

    Flipping a prefix or a suffix of a contiguous tensor reverses the order of
    fixed-size blocks, and (for a suffix) each block internally.  Both are the
    same affine map, so one kernel covers them; ``R == 1`` collapses to the
    plain whole-tensor reversal ``src = base - i``.
    """
    rank = A.ndim
    shape = A.shape
    # A flipped dimension of extent 1 is a no-op, so it cannot break a run.
    eff = [bool(flip_dims[i]) and shape[i] > 1 for i in range(rank)]
    idx = [i for i, on in enumerate(eff) if on]
    if not idx:
        # Nothing to reverse: torch.flip returns a fresh contiguous copy.
        return None
    if not A.is_contiguous():
        return _GENERIC

    is_prefix = idx == list(range(idx[-1] + 1))
    is_suffix = idx == list(range(idx[0], rank))
    if is_prefix and is_suffix:
        # Both hold only when every dimension is flipped, which is the same
        # whole-tensor reversal either way.
        is_prefix = True
    elif is_suffix:
        is_prefix = False
    elif not is_prefix:
        return _GENERIC

    if is_prefix:
        split = idx[-1] + 1
        block = math.prod(shape[split:])
        blocks = A.numel() // block
        return (blocks - 1) * block, -block, block, False

    split = idx[0]
    block = math.prod(shape[split:])
    return 0, block, block, True


def _launch_affine(A: torch.Tensor, out: torch.Tensor, base, step, block, reversed_row):
    n = A.numel()
    tile = _LARGE_BLOCK if n > _SMALL_LIMIT else _SMALL_BLOCK
    tile = min(tile, triton.next_power_of_2(n))
    num_warps = 1 if tile >= _LARGE_BLOCK else 2
    grid = (triton.cdiv(n, tile),)
    with torch_device_fn.device(A.device):
        _flip_affine_kernel[grid](
            A,
            out,
            n,
            base,
            step,
            R=block,
            REVERSED=reversed_row,
            BLOCK=tile,
            USE_I64=n > _INT32_MAX or abs(base) > _INT32_MAX,
            num_warps=num_warps,
        )


def _launch_meta(A: torch.Tensor, out: torch.Tensor, flip_dims):
    n = A.numel()
    ndim = A.ndim
    # Build on the host: a device-side tensor build dispatches through
    # ``lift_fresh`` under GEMS and costs a JIT compile per call.
    meta = torch.tensor(
        tuple(A.shape) + tuple(A.stride()) + tuple(flip_dims),
        dtype=torch.int64,
    ).to(A.device)
    tile = min(_LARGE_BLOCK, triton.next_power_of_2(n))
    with torch_device_fn.device(A.device):
        _flip_meta_kernel[(triton.cdiv(n, tile),)](
            A,
            out,
            meta,
            n,
            NDIM=ndim,
            BLOCK=tile,
            num_warps=4,
        )


def _flip_real(A: torch.Tensor, flip_dims) -> torch.Tensor:
    out = torch.empty_like(A, memory_format=torch.contiguous_format)
    if A.numel() == 0:
        return out

    plan = _plan(A, flip_dims)
    if plan is None:
        # Nothing to reverse; ``empty_like`` already produced the result.
        out.copy_(A)
        return out
    if plan is _GENERIC:
        _launch_meta(A, out, flip_dims)
        return out

    _launch_affine(A, out, *plan)
    return out


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN FLIP")
    flip_dims = [False] * A.ndim
    for dim in dims:
        if dim < -A.ndim or dim >= A.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-A.ndim}, {A.ndim - 1}], but got {dim})"
            )
        dim %= A.ndim
        if flip_dims[dim]:
            raise RuntimeError(f"dim {dim} appears multiple times in the list of dims")
        flip_dims[dim] = True

    if A.ndim == 0 or A.numel() <= 1 or not any(flip_dims):
        return A.clone()

    if A.is_complex():
        real_view = torch.view_as_real(A.resolve_conj())
        flipped = _flip_real(real_view, tuple(flip_dims) + (False,))
        return torch.view_as_complex(flipped)
    return _flip_real(A, tuple(flip_dims))
