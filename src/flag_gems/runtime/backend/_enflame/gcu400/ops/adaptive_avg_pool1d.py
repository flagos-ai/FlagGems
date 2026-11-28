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

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_avg_pool1d_uniform_kernel(
    X,
    Y,
    TOTAL,
    INV_W,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # Fast path for L_IN == L_OUT * W (exact division). Every pooling window
    # is then the contiguous slice [g*W, (g+1)*W) of the flattened input, and
    # the flat window index g doubles as the flat output index. The 2D tile
    # below is fully affine (row stride W, unit column stride), so it takes
    # the vectorized tar_load path instead of the per-element gather that the
    # runtime-div bin edges of the general kernel force.
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, W_PAD)
    row_valid = rows < TOTAL
    offs = rows[:, None] * W + cols[None, :]
    if W == W_PAD:
        valid = row_valid[:, None]
    else:
        valid = row_valid[:, None] & (cols < W)[None, :]
    values = tl.load(X + offs, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(values, 1) * INV_W
    tl.store(Y + rows, mean.to(Y.dtype.element_ty), mask=row_valid)


@libentry()
@triton.jit
def _adaptive_avg_pool1d_kernel(
    X,
    Y,
    L_IN,
    L_OUT,
    BLOCKS_PER_PLANE,
    UNITS,
    MAX_WINDOW: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    # One program handles BLOCK_OUT output positions of a single (N, C) plane.
    # aten's bin edges are floor(i*L_IN/L_OUT) and ceil((i+1)*L_IN/L_OUT);
    # loading a MAX_WINDOW-wide 2D tile lets all bins reduce in one tl.sum.
    # GCU400 caps grid.x at 65535 while N*C*blocks can be far larger, so the
    # grid is capped at one wave and programs walk units with a stride loop
    # (the trip-1 loop body folds away when every program owns one unit).
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for unit in tl.range(pid, UNITS, num_progs):
        out_block = unit % BLOCKS_PER_PLANE
        plane = unit // BLOCKS_PER_PLANE
        out_pos = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        bin_start = (out_pos * L_IN) // L_OUT
        bin_end = ((out_pos + 1) * L_IN + L_OUT - 1) // L_OUT
        window = tl.arange(0, MAX_WINDOW)
        in_pos = bin_start[:, None] + window[None, :]
        valid = (
            (out_pos[:, None] < L_OUT) & (in_pos < bin_end[:, None]) & (in_pos < L_IN)
        )
        values = tl.load(X + plane * L_IN + in_pos, mask=valid, other=0.0).to(
            tl.float32
        )
        count = tl.maximum(bin_end - bin_start, 1)
        mean = tl.sum(values, 1) / count.to(tl.float32)
        tl.store(
            Y + plane * L_OUT + out_pos,
            mean.to(Y.dtype.element_ty),
            mask=out_pos < L_OUT,
        )


def adaptive_avg_pool1d(input: torch.Tensor, output_size):
    """Adaptive average pooling over 1D input.

    The shared ``adaptive_avg_pool2d`` path lowers through
    ``_adaptive_pool_windows``, whose pointer-array tile makes the GCU400
    PassManager fail for every dtype. Calling ``aten::adaptive_avg_pool1d``
    directly is not an option either: FlagGems registers that very overload, so
    it recurses back here. This backend-specific kernel computes the bins
    directly from the 1D layout instead.

    When ``L_IN`` divides evenly into ``L_OUT`` bins (``L_IN % L_OUT == 0``)
    the windows tile the input contiguously, so a flat affine tile over the
    global window index replaces the runtime-div bin edges; that arm also
    keeps the grid at a single wave instead of one CTA per (plane, block).

    The log tag carries the vendor prefix: the accuracy test resolves it from
    the module path (``gcu400.ops.adaptive_avg_pool1d``), and ``gcu400`` is a
    known arch name, so ``GEMS`` alone does not match ``GEMS_ENFLAME``.
    """
    logger.debug("GEMS_ENFLAME ADAPTIVE_AVG_POOL1D")

    assert input.ndim == 3, f"adaptive_avg_pool1d expects 3D input, got {input.ndim}D"
    if isinstance(output_size, (list, tuple)):
        output_size = output_size[0]

    if not input.is_contiguous():
        input = input.contiguous()
    num_planes, channels, length_in = input.shape
    length_out = output_size
    out = torch.empty(
        (num_planes, channels, length_out),
        dtype=input.dtype,
        device=input.device,
    )
    if length_out == 0:
        return out

    total_out = num_planes * channels * length_out
    if length_in > 0 and length_in % length_out == 0 and input.numel() < 2**31:
        window = length_in // length_out
        window_pad = max(2, triton.next_power_of_2(window))
        # Target ~4096-element tiles (measured GCU400 amortization sweet
        # spot), keep the grid inside one <=48-CTA wave, and never exceed a
        # 64K-element tile so the fp32 staging stays well inside the DSM
        # budget even for very wide windows.
        block_rows = triton.next_power_of_2(
            max(16, triton.cdiv(total_out, 48), 4096 // window_pad)
        )
        block_rows = min(block_rows, triton.next_power_of_2(total_out))
        while block_rows * window_pad > 65536 and block_rows > 1:
            block_rows //= 2
        grid = (triton.cdiv(total_out, block_rows),)
        tile = block_rows * window_pad
        if tile <= 4096:
            num_warps = 1
        elif tile <= 16384:
            num_warps = 2
        else:
            num_warps = 4
        _adaptive_avg_pool1d_uniform_kernel[grid](
            input,
            out,
            total_out,
            1.0 / window,
            window,
            window_pad,
            block_rows,
            num_warps=num_warps,
        )
        return out

    max_window = triton.next_power_of_2(triton.cdiv(length_in, length_out) + 1)
    block_out = min(64, triton.next_power_of_2(length_out))
    blocks_per_plane = triton.cdiv(length_out, block_out)
    units = blocks_per_plane * num_planes * channels
    grid = (min(units, 48),)
    _adaptive_avg_pool1d_kernel[grid](
        input,
        out,
        length_in,
        length_out,
        blocks_per_plane,
        units,
        max_window,
        block_out,
        num_warps=1,
    )
    return out
