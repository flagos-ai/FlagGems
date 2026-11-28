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

"""GeGLU forward / dGeGLU backward, GCU400-optimized.

Both ops are memory-bound pointwise maps over a contiguous ``(M, 2H)`` input
whose two halves ``a = input[..., :H]`` and ``b = input[..., H:]`` are combined
elementwise, so the gradient also lands in the same ``(M, 2H)`` layout.

Design notes for GCU400:

* ``a`` and ``b`` of one row are two contiguous slices separated by ``H``
  elements, so a tile of ``ROWS`` rows x ``BLOCK`` columns reads
  ``[row*2H + col, ... + BLOCK)`` and the same plus ``H``. One CTA handles such
  a tile; the row index only needs one division per tile.
* GCU400 vector units prefer few warps with long per-thread vectors, so the
  tile uses ``num_warps=1`` and a large tile (``ROWS*BLOCK`` around 32K
  elements, bounded by the DSM budget). Measured on GCU400 this is ~20x faster
  than the public 64x64 2-D tiling, which caps ``grid.y`` at 255 and issues only
  4096 masked lanes per program.
* ``tanh`` is replaced by an equivalent single ``sigmoid``:
  ``0.5*x*(1+tanh(u)) == x*sigmoid(2u)`` with ``2u = 1.5957691*x*(1+0.044715x^2)``.
  The math is identical, but one transcendental per element instead of two.

The persistent 1-D loop keeps the grid at a few waves (``_GRID_CAP``): on GCU400
more CTAs do not add throughput here but do add per-CTA spawn cost.
"""

import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Resident programs. Three waves on the 16-SIP GCU400 configuration.
_GRID_CAP = 48
# Target lanes per tile; the real tile is ROWS x BLOCK_COL, which is bounded by
# the 917504-byte DSM budget (roughly 10 bytes per lane for these kernels).
_TILE_ELEMS = 32768
# Columns per tile. 64 halves the number of column blocks for wide rows while
# staying inside the DSM budget.
_MAX_BLOCK_COL = 64
# Upper bound on rows per tile.
_MAX_ROWS = 2048
_NUM_WARPS = 1
_INT32_SAFE_LIMIT = 0x7FFFFFFF


@libentry()
@triton.jit(do_not_specialize=["M", "H", "num_cb"])
def _geglu_forward_kernel(
    inp, out, M, H, num_cb, ROWS: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    total_blocks = tl.cdiv(M, ROWS) * num_cb
    offs_r = tl.arange(0, ROWS)
    offs_c = tl.arange(0, BLOCK)
    two_h = 2 * H
    for block_id in tl.range(pid, total_blocks, num_pids):
        row_group = block_id // num_cb
        col_block = block_id - row_group * num_cb
        row = row_group * ROWS + offs_r
        col = col_block * BLOCK + offs_c
        in_off = row[:, None] * two_h + col[None, :]
        mask = (col[None, :] < H) & (row[:, None] < M)
        x_a = tl.load(inp + in_off, mask=mask, other=0.0).to(tl.float32)
        x_b = tl.load(inp + in_off + H, mask=mask, other=0.0).to(tl.float32)
        # 2 * 0.79788456, 0.044715: the tanh-GELU constants doubled so that
        # 0.5*x*(1+tanh(u)) can be written as x*sigmoid(2u).
        u = 1.5957691216057308 * x_a * (1.0 + 0.044715 * x_a * x_a)
        y = x_a * tl.sigmoid(u) * x_b
        tl.store(out + row[:, None] * H + col[None, :], y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["M", "H", "num_cb"])
def _dgeglu_backward_kernel(
    grad_out, inp, grad_in, M, H, num_cb, ROWS: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    total_blocks = tl.cdiv(M, ROWS) * num_cb
    offs_r = tl.arange(0, ROWS)
    offs_c = tl.arange(0, BLOCK)
    two_h = 2 * H
    for block_id in tl.range(pid, total_blocks, num_pids):
        row_group = block_id // num_cb
        col_block = block_id - row_group * num_cb
        row = row_group * ROWS + offs_r
        col = col_block * BLOCK + offs_c
        in_off = row[:, None] * two_h + col[None, :]
        out_off = row[:, None] * H + col[None, :]
        mask = (col[None, :] < H) & (row[:, None] < M)
        x_a = tl.load(inp + in_off, mask=mask, other=0.0).to(tl.float32)
        x_b = tl.load(inp + in_off + H, mask=mask, other=0.0).to(tl.float32)
        g_out = tl.load(grad_out + out_off, mask=mask, other=0.0).to(tl.float32)
        u = 1.5957691216057308 * x_a * (1.0 + 0.044715 * x_a * x_a)
        sig = tl.sigmoid(u)
        gelu_a = x_a * sig
        # d/dx [x*sigmoid(u)] = sigmoid(u) + x*sigmoid(u)*(1-sigmoid(u))*du/dx
        d_gelu = sig + gelu_a * (1.0 - sig) * 1.5957691216057308 * (
            1.0 + 3.0 * 0.044715 * x_a * x_a
        )
        tl.store(grad_in + in_off, g_out * x_b * d_gelu, mask=mask)
        tl.store(grad_in + in_off + H, g_out * gelu_a, mask=mask)


def _tile_shape(half, rows_total):
    """Pick (ROWS, BLOCK, num_col_blocks) for one op call."""
    block = min(_MAX_BLOCK_COL, triton.next_power_of_2(half))
    rows_budget = min(_MAX_ROWS, max(1, _TILE_ELEMS // block))
    rows = min(rows_budget, triton.next_power_of_2(rows_total))
    num_col_blocks = triton.cdiv(half, block)
    return rows, block, num_col_blocks


def _launch(kernel, args, rows, half, num_col_blocks, num_rows, block, device):
    total_blocks = triton.cdiv(num_rows, rows) * num_col_blocks
    grid = (min(total_blocks, _GRID_CAP),)
    with torch_device_fn.device(device):
        kernel[grid](
            *args,
            num_rows,
            half,
            num_col_blocks,
            ROWS=rows,
            BLOCK=block,
            num_warps=_NUM_WARPS,
        )


def geglu(input_tensor: torch.Tensor, quantizer: Optional[Any] = None) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME GEGLU")
    shape = input_tensor.shape
    half = shape[-1] // 2
    if half == 0:
        return torch.empty(
            (*shape[:-1], half), device=input_tensor.device, dtype=input_tensor.dtype
        )

    inp = input_tensor.contiguous()
    num_rows = inp.numel() // (2 * half)
    out = torch.empty((num_rows, half), device=inp.device, dtype=inp.dtype)

    rows, block, num_col_blocks = _tile_shape(half, num_rows)
    num_blocks = triton.cdiv(num_rows, rows) * num_col_blocks
    if num_blocks * rows * block > _INT32_SAFE_LIMIT:
        from flag_gems.fused.geglu import geglu as _public_geglu

        return _public_geglu(input_tensor, quantizer)

    _launch(
        _geglu_forward_kernel,
        (inp, out),
        rows,
        half,
        num_col_blocks,
        num_rows,
        block,
        inp.device,
    )
    return out.view(*shape[:-1], half)


def dgeglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME DGEGLU")
    shape = input_tensor.shape
    half = shape[-1] // 2
    if half == 0:
        return torch.empty_like(input_tensor)

    inp = input_tensor.contiguous()
    num_rows = inp.numel() // (2 * half)
    grad_out = grad_output.contiguous().view(num_rows, half)
    grad_in = torch.empty_like(inp)

    rows, block, num_col_blocks = _tile_shape(half, num_rows)
    num_blocks = triton.cdiv(num_rows, rows) * num_col_blocks
    if num_blocks * rows * block > _INT32_SAFE_LIMIT:
        from flag_gems.fused.geglu import dgeglu as _public_dgeglu

        return _public_dgeglu(grad_output, input_tensor, quantizer)

    _launch(
        _dgeglu_backward_kernel,
        (grad_out, inp, grad_in),
        rows,
        half,
        num_col_blocks,
        num_rows,
        block,
        inp.device,
    )
    return grad_in.view_as(input_tensor)
