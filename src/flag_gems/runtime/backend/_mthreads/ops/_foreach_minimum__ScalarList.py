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
"""_foreach_minimum__ScalarList implemented with Triton kernels.

Semantics (ATen reference):
    for i in range(len(self)):
        self[i] <- minimum(self[i], self[i].dtype(scalars[i]))   # in place
    returns None

minimum(a, b) is NaN-propagating and returns b on value ties:
    min(a, b) = isnan(a) ? a : (a < b ? a : b)
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_binary import (
    _foreach_minimum__ScalarList as default__foreach_minimum__ScalarList,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_MAX_STRIDE_DIMS = 8
_MAX_FUSED_SLOTS = 4
# Multi-slot fusion only pays for launch-bound lists: on large lists the
# 2D-grid form loses ~8-14% per-element efficiency versus one flat launch per
# tensor, which costs more than the three saved launches.
_FUSED_MAX_NUMEL = 1 << 20
# Elements handled by one program (BLOCK * VEC).
_TILE = 1024


@triton.jit
def _min_body(x_ptr, n_elem, scalar, pid, BLOCK: tl.constexpr, VEC: tl.constexpr):
    # Each program owns BLOCK granules of VEC adjacent elements; the granule
    # shape makes the fast axis a full 16-byte vector, so both the load and
    # the store stay vectorized.
    offs = (
        pid * (BLOCK * VEC)
        + tl.arange(0, BLOCK)[:, None] * VEC
        + tl.arange(0, VEC)[None, :]
    )
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask)
    sv = scalar.to(x.dtype)
    r = tl.where(x != x, x, tl.where(x < sv, x, sv))
    # Write only the granules that hold at least one changed element: an
    # unchanged element already holds the result, so skipping its store leaves
    # memory bit-identical while dropping most of the write traffic. NaN lanes
    # satisfy r != x and are rewritten unchanged. The predicate is constant
    # along the vector axis, so each granule costs one predicated vector store
    # instead of one masked store per element.
    changed = tl.max((r != x).to(tl.int32), axis=1) != 0
    tl.store(x_ptr + offs, r, mask=mask & changed[:, None])


@triton.jit
def _min_scalar_flat_kernel(
    x_ptr, n_elem, scalar, BLOCK: tl.constexpr, VEC: tl.constexpr
):
    _min_body(x_ptr, n_elem, scalar, tl.program_id(0), BLOCK, VEC)


@triton.jit
def _min_scalar_list_kernel(
    x0,
    x1,
    x2,
    x3,
    n0,
    n1,
    n2,
    n3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """One launch for a whole (same-dtype, dense) tensor list.

    Axis 0 walks blocks of elements, axis 1 selects the list slot so every
    slot keeps its own pointer, length and scalar. Trailing slots of a short
    list are never scheduled because the grid's second extent equals the
    number of active slots.
    """
    slot = tl.program_id(1)
    pid = tl.program_id(0)
    if slot == 0:
        _min_body(x0, n0, s0, pid, BLOCK, VEC)
    if slot == 1:
        _min_body(x1, n1, s1, pid, BLOCK, VEC)
    if slot == 2:
        _min_body(x2, n2, s2, pid, BLOCK, VEC)
    if slot == 3:
        _min_body(x3, n3, s3, pid, BLOCK, VEC)


@triton.jit
def _min_scalar_strided_kernel(
    x_ptr,
    n_elem,
    scalar,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elem
    rem = idx.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)

    d = d0.to(tl.int64)
    i = rem % d
    rem = rem // d
    off += i * s0.to(tl.int64)
    d = d1.to(tl.int64)
    i = rem % d
    rem = rem // d
    off += i * s1.to(tl.int64)
    d = d2.to(tl.int64)
    i = rem % d
    rem = rem // d
    off += i * s2.to(tl.int64)
    d = d3.to(tl.int64)
    i = rem % d
    rem = rem // d
    off += i * s3.to(tl.int64)
    d = d4.to(tl.int64)
    i = rem % d
    rem = rem // d
    off += i * s4.to(tl.int64)
    d = d5.to(tl.int64)
    i = rem % d
    rem = rem // d
    off += i * s5.to(tl.int64)
    d = d6.to(tl.int64)
    i = rem % d
    rem = rem // d
    off += i * s6.to(tl.int64)
    d = d7.to(tl.int64)
    i = rem % d
    off += i * s7.to(tl.int64)

    x = tl.load(x_ptr + off, mask=mask)
    sv = scalar.to(x.dtype)
    tl.store(x_ptr + off, tl.where(x != x, x, tl.where(x < sv, x, sv)), mask=mask)


def _is_flat_dense(x):
    """True when a flat traversal from data_ptr covers the storage exactly."""
    if x.is_contiguous():
        return True
    shape = x.shape
    strides = x.stride()
    dims = sorted((strides[i], shape[i]) for i in range(len(shape)) if shape[i] != 1)
    expected = 1
    for stride, size in dims:
        if stride != expected:
            return False
        expected *= size
    return True


def _collapse_dims(shape, strides):
    """Collapse adjacent contiguous dimensions (innermost first)."""
    dims = []
    for size, stride in zip(shape, strides):
        if size != 1:
            dims.append([size, stride])
    dims.reverse()
    collapsed = []
    for size, stride in dims:
        if collapsed and collapsed[-1][1] == stride * size:
            collapsed[-1][0] *= size
            collapsed[-1][1] = stride
        else:
            collapsed.append([size, stride])
    return collapsed


def _launch_strided(x, n_elem, scalar, block):
    dims = _collapse_dims(x.shape, x.stride())
    if len(dims) > _MAX_STRIDE_DIMS:
        raise RuntimeError(
            "_foreach_minimum__ScalarList: unsupported tensor layout with "
            f"{len(dims)} non-collapsible dimensions"
        )
    sizes = [1] * _MAX_STRIDE_DIMS
    steps = [0] * _MAX_STRIDE_DIMS
    for k, (size, stride) in enumerate(dims):
        sizes[k] = size
        steps[k] = stride
    grid = ((n_elem + block - 1) // block,)
    _min_scalar_strided_kernel[grid](
        x,
        n_elem,
        scalar,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        steps[0],
        steps[1],
        steps[2],
        steps[3],
        steps[4],
        steps[5],
        steps[6],
        steps[7],
        BLOCK=block,
        num_warps=4,
    )


_FAST = {}


def _dispatch(jit_fn, grid, args, const, num_warps, key):
    """Launch ``jit_fn``, replaying the compiled variant once it exists.

    Triton's Python-level launch costs ~20 us on this target, which is the whole
    budget for the short lists this op sees, so the CompiledKernel is replayed
    directly. The ordinary launch stays as the fallback.
    """
    entry = _FAST.get(key)
    if entry is not None:
        compiled, stream = entry
        try:
            compiled.run(
                grid[0],
                grid[1] if len(grid) > 1 else 1,
                1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                *const.values(),
            )
            return
        except Exception:
            _FAST.pop(key, None)
    compiled = jit_fn[grid](*args, num_warps=num_warps, **const)
    try:
        drv = _driver.active
        _FAST[key] = (compiled, drv.get_current_stream(drv.get_current_device()))
    except Exception:
        _FAST.pop(key, None)


def _foreach_minimum__ScalarList(self, scalars):
    logger.debug("GEMS_MTHREADS _FOREACH_MINIMUM__SCALARLIST")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in self
    ):
        return default__foreach_minimum__ScalarList(self, scalars)
    tensors = self
    n_tensors = len(tensors)
    if n_tensors == 0:
        return None

    if isinstance(scalars, torch.Tensor) and scalars.dim() == 0:
        scalars = [scalars] * n_tensors
    elif isinstance(scalars, (int, float)):
        scalars = [scalars] * n_tensors
    else:
        scalars = list(scalars)
        if len(scalars) != n_tensors:
            raise RuntimeError(
                "_foreach_minimum__ScalarList: expected the tensor list and the "
                f"scalar list to have the same length, but got {n_tensors} and "
                f"{len(scalars)}"
            )

    dtype = tensors[0].dtype
    fused = True
    numels = None
    if fused:
        for x in tensors:
            if x.dtype is not dtype or not _is_flat_dense(x):
                fused = False
                break
        if fused:
            numels = [x.numel() for x in tensors]
            if max(numels) > _FUSED_MAX_NUMEL:
                fused = False

    if fused:
        vec = 16 // tensors[0].element_size()
        # The fused kernel has four slots; a longer list goes out as several
        # four-wide launches rather than one launch per tensor.
        for start in range(0, n_tensors, _MAX_FUSED_SLOTS):
            chunk_t = tensors[start : start + _MAX_FUSED_SLOTS]
            chunk_n = numels[start : start + _MAX_FUSED_SLOTS]
            chunk_v = scalars[start : start + _MAX_FUSED_SLOTS]
            k = len(chunk_t)
            max_numel = max(chunk_n)
            if max_numel == 0:
                continue
            pad = _MAX_FUSED_SLOTS - k
            ptrs = list(chunk_t) + [chunk_t[0]] * pad
            lens = list(chunk_n) + [0] * pad
            vals = [float(v) for v in chunk_v] + [0.0] * pad
            grid = ((max_numel + _TILE - 1) // _TILE, k)
            args = (
                ptrs[0],
                ptrs[1],
                ptrs[2],
                ptrs[3],
                lens[0],
                lens[1],
                lens[2],
                lens[3],
                vals[0],
                vals[1],
                vals[2],
                vals[3],
            )
            key = (
                "list",
                chunk_t[0].dtype,
                k,
                tuple(lens),
                tuple(p.data_ptr() & 15 for p in ptrs),
                _TILE // vec,
            )
            _dispatch(
                _min_scalar_list_kernel,
                grid,
                args,
                {"BLOCK": _TILE // vec, "VEC": vec},
                4,
                key,
            )
        return None

    for x, scalar in zip(tensors, scalars):
        n_elem = x.numel()
        if n_elem == 0:
            continue
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.item()
        scalar = float(scalar)
        if _is_flat_dense(x):
            vec = 16 // x.element_size()
            grid = ((n_elem + _TILE - 1) // _TILE,)
            _min_scalar_flat_kernel[grid](
                x, n_elem, scalar, BLOCK=_TILE // vec, VEC=vec, num_warps=4
            )
        else:
            _launch_strided(x, n_elem, scalar, _TILE)
    return None
