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
#
# GCU400 customization (2026-11-21 retune): DMA bytes per tile (not CTAs or
# num_stages) set the streaming bandwidth. Measured A/B on do_bench(median):
# 512 KiB/tile at num_warps=4 reaches ~1.73 TB/s for fp32 and ~2.04 TB/s for
# 16-bit in the big band, versus ~1.5 TB/s for the previous 256 KiB ladder;
# the 1M-64M fp32 band prefers 8-byte lanes with a 64 KiB tile.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs); see ops/deg2rad.py for the grid.x limit
# rationale.
_GRID_SIZE = 48

# --- TUNING PARAMS START ---
_TINY_N = 8192
_SMALL_N = 1 << 20
_LARGE_N = 1 << 26
_LARGE_TILE_BYTES = 512 * 1024
_MEDIUM_TILE_BYTES = 256 * 1024
_SMALL_TILE_BYTES = 32 * 1024
_MEDIUM_WIDE_TILE_BYTES = 64 * 1024
# --- TUNING PARAMS END ---


@triton.jit(do_not_specialize=["n_elements"])
def view_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_ctas = tl.num_programs(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    lane_offset = tl.arange(0, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tile, num_ctas):
        offsets = tile_id * BLOCK_SIZE + lane_offset
        if MASKED:
            mask = offsets < n_elements
            tl.store(
                dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask
            )
        else:
            tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


def _pick_launch(n, element_size):
    """Return (view_bytes, tile_bytes, num_warps) for this element count/width."""
    if n <= _TINY_N:
        return element_size, max(triton.next_power_of_2(n), 64) * element_size, 4
    if n <= _SMALL_N:
        return element_size, _SMALL_TILE_BYTES, 4
    if n >= _LARGE_N:
        return element_size, _LARGE_TILE_BYTES, 4
    if element_size == 4:
        # Medium fp32 band: 8-byte lanes keep the DMA queue full.
        return 8, _MEDIUM_WIDE_TILE_BYTES, 4
    return element_size, _MEDIUM_TILE_BYTES, 2


def _widen(t, view_bytes):
    """Reinterpret a contiguous tensor with wider lanes (pure metadata view).

    The flat view is taken first: ``Tensor.view(dtype)`` alone requires the last
    dimension to divide evenly, which fails for shapes such as 16x7x57x32x29 even
    though the total element count is even.
    """
    if view_bytes == 8:
        return t.view(-1).view(torch.int64) if t.element_size() == 4 else None
    if view_bytes == 4:
        return t.view(-1).view(torch.int32) if t.element_size() == 2 else None
    return None


def view_copy(x: torch.Tensor, size) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME VIEW_COPY")
    """Wrapper for aten::view_copy: copy of `x` with the specified shape."""
    # Handle SymInt[] - convert to tuple of ints
    if isinstance(size, torch.SymInt):
        size = (int(size),)
    elif isinstance(size, (list, tuple)):
        size = tuple(int(s) if isinstance(s, torch.SymInt) else s for s in size)

    n_elements = x.numel()

    # Handle -1 (infer this dimension)
    if -1 in size:
        if size.count(-1) > 1:
            raise RuntimeError(f"view_copy: only one dimension can be -1, got {size}")
        target_numel_except_minus1 = 1
        for s in size:
            if s != -1:
                target_numel_except_minus1 *= s
        inferred_dim = n_elements // target_numel_except_minus1
        size = tuple(inferred_dim if s == -1 else s for s in size)

    # Validate total number of elements matches
    target_numel = 1
    for s in size:
        target_numel *= s
    if n_elements != target_numel:
        raise RuntimeError(
            f"view_copy: cannot reshape tensor of size {n_elements} into shape {size}"
        )

    if n_elements == 0:
        return torch.empty(size, dtype=x.dtype, device=x.device)

    out = torch.empty(size, dtype=x.dtype, device=x.device)
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()

    view_bytes, tile_bytes, num_warps = _pick_launch(n_elements, x.element_size())
    src_view, dst_view = src, out
    if view_bytes > x.element_size():
        wide_src, wide_dst = _widen(src, view_bytes), _widen(out, view_bytes)
        legal = (
            wide_src is not None
            and wide_dst is not None
            and src.data_ptr() % view_bytes == 0
            and out.data_ptr() % view_bytes == 0
        )
        if legal:
            src_view, dst_view = wide_src, wide_dst
        else:
            tile_bytes, num_warps = _MEDIUM_TILE_BYTES, 2
    block_size = max(tile_bytes // src_view.element_size(), 64)
    n_lanes = src_view.numel()
    num_tile = triton.cdiv(n_lanes, block_size)
    grid = (min(_GRID_SIZE, num_tile), 1, 1)
    with torch_device_fn.device(x.device.index):
        view_copy_kernel[grid](
            src_view,
            dst_view,
            n_lanes,
            BLOCK_SIZE=block_size,
            MASKED=bool(n_lanes % block_size),
            num_warps=num_warps,
        )
    return out
