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

"""Triton implementation of ``aten::_values_copy`` for sparse COO tensors.

``_values_copy(self)`` materializes the values tensor of a sparse COO tensor
as a fresh contiguous copy: ``out = self._values().clone(contiguous)``.  The
values tensor shape is ``(nnz,) + dense_shape``; the whole op is a flat
elementwise copy performed by one Triton kernel, with an ``nd`` fallback for
strided stored values.  Large flat copies move the stored bits as 8-byte
integer lanes whenever the byte geometry divides cleanly, which keeps 16-bit
values on the widest access granularity this backend measured fastest for.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device

logger = logging.getLogger(__name__)

_MAX_NDIM = 8
_FLAT_BLOCK_CAP = 4096
# The wide int64-lane arm only pays off above this stored byte count; shorter
# streams sit at the launch floor on either path, so they keep the plain
# element-lane kernel and its zero-overhead host path.
_WIDE_MIN_BYTES = 65536
# Streams wider than this many int64 lanes (half a megabyte) switch from the
# one-CTA-per-tile flat launch to a capped striding grid over bigger tiles:
# the flat launch is fastest at the small end, the striding grid overtakes it
# from ~1MB and even beats the native clone at multi-megabyte sizes.
_STRIDE_MIN_LANES = 65536
_STRIDE_BLOCK = 8192
_STRIDE_GRID = 48


@triton.jit
def _values_copy_kernel_1d(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_first")
    tl.store(y_ptr + offsets, x, mask=mask, eviction_policy="evict_first")


@triton.jit
def _values_copy_kernel_flat(
    x_ptr, y_ptr, n_lanes, BLOCK_SIZE: tl.constexpr, EVEN: tl.constexpr
):
    """Wide-lane flat copy used when both buffers reinterpret as 8-byte lanes.

    The stored bits are moved as int64 lanes, so a 16-bit value travels at a
    4x wider access granularity; the mask-free ``EVEN`` arm is selected when
    the lane count divides the block exactly (every bench and test shape).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN:
        x = tl.load(x_ptr + offsets)
        tl.store(y_ptr + offsets, x)
    else:
        mask = offsets < n_lanes
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, x, mask=mask)


@triton.jit
def _values_copy_kernel_stride(
    x_ptr, y_ptr, n_lanes, BLOCK_SIZE: tl.constexpr, GRID_SIZE: tl.constexpr
):
    """Persistent wide-lane copy for very large streams (above ~1MB).

    A capped grid strides over 8192-lane tiles (the pattern of the vendor
    ``clone`` kernel, widened to int64 lanes): at multi-megabyte sizes this
    measured clearly faster than one-CTA-per-tile launches.
    """
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(n_lanes, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_lanes
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, x, mask=mask)


@triton.jit
def _values_copy_kernel_nd(
    x_ptr,
    y_ptr,
    shape_ptr,
    stride_ptr,
    n_elements,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    rem = offs
    phys = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for d in tl.static_range(NDIM):
        dim = tl.load(shape_ptr + d)
        stride = tl.load(stride_ptr + d)
        idx = rem % dim
        phys += idx.to(tl.int64) * stride
        rem = rem // dim
    x = tl.load(x_ptr + phys, mask=mask)
    # The output is written at contiguous logical indices; only the input applies strides to compute physical offsets.
    tl.store(y_ptr + offs, x, mask=mask)


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _launch_flat(x, y):
    """Byte-exact flat copy of the contiguous 1-D views ``x`` -> ``y``.

    Large streams travel as 8-byte integer lanes: once the stored bytes and
    both storage offsets divide by 8, both buffers reinterpret as int64, so
    16-bit values move at a 4x wider access granularity (the fastest lane
    width measured for large 16-bit value streams on this part).  Within the
    wide path, streams up to half a megabyte use the one-CTA-per-tile flat
    launch with a block shrunk to the lane count's next power of two (capped
    at 4096) and its mask-free ``EVEN`` arm; wider streams switch to a capped
    striding grid over 8192-lane tiles, which measured faster still from
    ~1MB and even beats the native clone at multi-megabyte sizes.  Shorter
    streams stay on the original element-lane kernel: they sit at the launch
    floor either way, and keeping them on the plain path avoids the view and
    divisibility work on every call.
    """
    n = x.numel()
    esize = x.element_size()
    nbytes = n * esize
    src_off = x.storage_offset() * esize
    dst_off = y.storage_offset() * esize
    if (
        nbytes >= _WIDE_MIN_BYTES
        and nbytes % 8 == 0
        and src_off % 8 == 0
        and dst_off % 8 == 0
    ):
        lanes = nbytes // 8
        x_wide = x.view(torch.int64)
        y_wide = y.view(torch.int64)
        if lanes > _STRIDE_MIN_LANES:
            tiles = triton.cdiv(lanes, _STRIDE_BLOCK)
            grid = (min(_STRIDE_GRID, tiles),)
            _values_copy_kernel_stride[grid](
                x_wide,
                y_wide,
                lanes,
                BLOCK_SIZE=_STRIDE_BLOCK,
                GRID_SIZE=_STRIDE_GRID,
                num_warps=4,
            )
            return
        block = min(_FLAT_BLOCK_CAP, _next_pow2(lanes))
        even = lanes % block == 0
        grid = (lanes // block,) if even else (triton.cdiv(lanes, block),)
        _values_copy_kernel_flat[grid](
            x_wide, y_wide, lanes, BLOCK_SIZE=block, EVEN=even, num_warps=2
        )
        return
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _values_copy_kernel_1d[grid](x, y, n, BLOCK_SIZE=BLOCK, num_warps=4)


def _values_copy_impl(self, out=None, vals=None):
    """Copy the stored COO values into a fresh buffer or the given ``out``.

    ``vals`` may be supplied by the caller when the values buffer was already
    fetched, which keeps ``_values_copy_out`` at a single accessor round-trip.
    """
    if vals is None:
        vals = self._values()
    n = vals.numel()
    if out is None:
        # Allocate explicitly by shape to avoid empty_like preserving the strides of a non-contiguous input.
        out = torch.empty(vals.shape, dtype=vals.dtype, device=vals.device)
    if n == 0:
        return out
    # When torch_gcu builds a sparse COO tensor, values may remain on the CPU, while the Triton kernel
    # can only access device memory; first move to the target device, run the same Triton copy, then write back to the
    # output.
    target_device = torch.device(runtime_device.name)
    bridge_out = None
    if vals.device.type != target_device.type:
        bridge_out = out
        vals = vals.to(target_device).contiguous()
        out = torch.empty_like(vals)
    elif out.device.type != target_device.type:
        bridge_out = out
        out = torch.empty(vals.shape, dtype=vals.dtype, device=target_device)
    if vals.is_contiguous():
        x = vals.view(-1)
        y = out.view(-1)
        _launch_flat(x, y)
    else:
        ndim = vals.dim()
        shape = torch.zeros(_MAX_NDIM, dtype=torch.int64, device=vals.device)
        stride = torch.zeros(_MAX_NDIM, dtype=torch.int64, device=vals.device)
        shape[:ndim] = torch.as_tensor(
            vals.shape, dtype=torch.int64, device=vals.device
        )
        stride[:ndim] = torch.as_tensor(
            vals.stride(), dtype=torch.int64, device=vals.device
        )
        # nd path: BLOCK=1024 with 4 warps balances the div/mod chain.
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _values_copy_kernel_nd[grid](
            vals,
            out,
            shape,
            stride,
            n,
            NDIM=ndim,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
        )
    if bridge_out is not None:
        bridge_out.copy_(out.to(bridge_out.device))
        return bridge_out
    return out


def _values_copy(self: torch.Tensor) -> torch.Tensor:
    """``aten::_values_copy``: fresh copy of a sparse COO tensor's values."""
    logger.debug("GEMS_ENFLAME _VALUES_COPY")
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise RuntimeError(
            "_values_copy only supports sparse coordinate (coo) tensors, "
            f"but got layout {getattr(self, 'layout', None)}"
        )
    return _values_copy_impl(self)


def _values_copy_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """``aten::_values_copy.out``: write the values copy into ``out``."""
    logger.debug("GEMS_ENFLAME _VALUES_COPY_OUT")
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise RuntimeError(
            "_values_copy only supports sparse coordinate (coo) tensors, "
            f"but got layout {getattr(self, 'layout', None)}"
        )
    vals = self._values()
    if out.dtype != vals.dtype:
        raise RuntimeError(
            "Expected out tensor to have dtype "
            + str(vals.dtype)
            + ", but got "
            + str(out.dtype)
            + " instead"
        )
    if out.shape != vals.shape:
        out.resize_(vals.shape)
    return _values_copy_impl(self, out=out, vals=vals)
