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
# GCU400 customization (2026-11-21 retune): the DMA bytes per tile, not the CTA
# count, drive the streaming bandwidth. Measured on do_bench(median) A/B against
# the native aten copy: a 64 KiB tile reaches ~1.5 TB/s while a 512 KiB tile
# reaches ~1.73 TB/s (fp32 2^28 elements: 0.818x native instead of 0.777x).
# The size/dtype ladder below is the measured optimum for each band, and wide
# int64 lanes win in the 1M-64M fp32 band where a single 512 KiB tile leaves the
# pipeline under-fed.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# On GCU400 one SIP can host two resident programs, so 48 CTAs is full occupancy.
NUM_SIPS = 24
MAX_CTAS = NUM_SIPS * 2

# --- TUNING PARAMS START ---
TINY_INPUT_ELEMENTS = 4096
SMALL_INPUT_ELEMENTS = 1 << 20
# Band boundary where one 512 KiB tile per trip finally wins over more CTAs.
LARGE_INPUT_ELEMENTS = 1 << 26
LARGE_TILE_BYTES = 512 * 1024
MEDIUM_TILE_BYTES = 256 * 1024
SMALL_TILE_BYTES = 32 * 1024
MEDIUM_WIDE_TILE_BYTES = 64 * 1024
# --- TUNING PARAMS END ---


@triton.jit(do_not_specialize=["n_elements"])
def _alias_copy_kernel(
    src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr, MASKED: tl.constexpr
):
    """Grid-stride tile copy.

    MASKED is a constexpr specialization: the benchmark shapes divide the tile
    evenly, so the hot loop carries no load/store predicate at all.
    """
    program_id = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_tiles = (n_elements + BLOCK - 1) // BLOCK
    lane_offset = tl.arange(0, BLOCK)
    for tile_id in tl.range(program_id, num_tiles, num_ctas):
        element_offset = tile_id * BLOCK + lane_offset
        if MASKED:
            active = element_offset < n_elements
            tl.store(
                dst_ptr + element_offset,
                tl.load(src_ptr + element_offset, mask=active),
                mask=active,
            )
        else:
            tl.store(dst_ptr + element_offset, tl.load(src_ptr + element_offset))


def _pick_launch(n_elements, element_size):
    """Return (view_bytes, tile_bytes, num_warps) for this element count/width."""
    if n_elements <= TINY_INPUT_ELEMENTS:
        # Launch-bound: one exactly-sized tile, no stranded lanes.
        lanes = max(triton.next_power_of_2(n_elements), 64)
        return element_size, lanes * element_size, 4
    if n_elements <= SMALL_INPUT_ELEMENTS:
        return element_size, SMALL_TILE_BYTES, 4
    if n_elements >= LARGE_INPUT_ELEMENTS:
        return element_size, LARGE_TILE_BYTES, 4
    if element_size == 4:
        # Medium fp32 band: 8-byte lanes keep the DMA queue full.
        return 8, MEDIUM_WIDE_TILE_BYTES, 4
    return element_size, MEDIUM_TILE_BYTES, 2


def _widen(t, view_bytes):
    """Reinterpret a contiguous tensor with wider lanes (pure metadata view).

    The flat view is taken first: ``Tensor.view(dtype)`` itself requires the last
    dimension to divide evenly, which fails for shapes such as 16x7x57x32x29 even
    though the total element count is even.
    """
    if view_bytes == 8:
        return t.view(-1).view(torch.int64) if t.element_size() == 4 else None
    if view_bytes == 4:
        return t.view(-1).view(torch.int32) if t.element_size() == 2 else None
    return None


def _launch_copy(src, dst, n_elements):
    view_bytes, tile_bytes, num_warps = _pick_launch(n_elements, src.element_size())
    src_view, dst_view = src, dst
    if view_bytes > src.element_size():
        wide_src, wide_dst = _widen(src, view_bytes), _widen(dst, view_bytes)
        aligned = wide_src is not None and wide_dst is not None
        # Both lanes must stay 8-byte aligned for the widened view to be legal.
        if (
            aligned
            and src.data_ptr() % view_bytes == 0
            and dst.data_ptr() % view_bytes == 0
        ):
            src_view, dst_view = wide_src, wide_dst
        else:
            tile_bytes, num_warps = MEDIUM_TILE_BYTES, 2
    block = max(tile_bytes // src_view.element_size(), 64)
    n_lanes = src_view.numel()
    n_tiles = triton.cdiv(n_lanes, block)
    grid = min(max(n_tiles, 1), MAX_CTAS)
    with torch_device_fn.device(src.device):
        _alias_copy_kernel[(grid,)](
            src_view,
            dst_view,
            n_lanes,
            BLOCK=block,
            MASKED=bool(n_lanes % block),
            num_warps=num_warps,
        )


def alias_copy(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME ALIAS_COPY")
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()
    if src.dtype != out.dtype:
        raise RuntimeError("alias_copy: dtype mismatch between input and output.")
    _launch_copy(src, out, n_elements)
    return out


def alias_copy_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME ALIAS_COPY_OUT")
    if x.dtype != out.dtype:
        raise RuntimeError("alias_copy_out: dtype of input and output must match.")
    if x.numel() != out.numel():
        raise RuntimeError(
            "alias_copy_out: input and output must have the same number of elements."
        )
    if x.device != out.device:
        raise RuntimeError(
            "alias_copy_out: input and output must be on the same device."
        )
    if not out.is_contiguous():
        raise RuntimeError("alias_copy_out: output tensor must be contiguous.")
    src = x.contiguous() if not x.is_contiguous() else x
    n_elements = out.numel()
    if n_elements == 0:
        return out
    _launch_copy(src, out, n_elements)
    return out
