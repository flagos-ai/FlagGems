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
# GCU300-specific reflection_pad1d.
# Hybrid strategy:
#   - Tiny shapes (< 200 elems): aten flip+copy (3 strided copies, low dispatch overhead)
#   - Medium+large shapes: single fused two-phase kernel (1 launch)
#     Phase 1: block_ptr DMA for center identity copy
#     Phase 2: element-wise gather for left+right reflection edges
# Two sequential phases in one kernel — DMA fast path is preserved because
# phases are NOT per-element branched. The edge gather runs after all DMA
# tiles complete, reusing the same SPs with zero extra launch overhead.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_TINY_THRESHOLD = 500_000


@triton.jit(
    do_not_specialize=[
        "B",
        "W_in",
        "W_out",
        "pad_left",
        "left_count",
        "right_start",
        "right_count",
        "edge_total",
        "n_tiles_b",
        "n_tiles_w",
    ]
)
def fused_two_phase_kernel(
    in_ptr,
    out_ptr,
    B,
    W_in,
    W_out,
    pad_left,
    left_count,
    right_start,
    right_count,
    edge_total,
    n_tiles_b,
    n_tiles_w,
    GRID_DIM: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
):
    """Single-launch fused kernel: Phase 1 center DMA, Phase 2 edge gather."""
    pid = tl.program_id(axis=0)
    total_center = n_tiles_b * n_tiles_w

    # Phase 1: Center DMA — block_ptr identity copy
    for tile_id in tl.range(pid, total_center, GRID_DIM):
        pid_b = tile_id // n_tiles_w
        pid_w = tile_id % n_tiles_w

        in_block = tl.make_block_ptr(
            base=in_ptr,
            shape=(B, W_in),
            strides=(W_in, 1),
            offsets=(pid_b * BLOCK_B, pid_w * BLOCK_W),
            block_shape=(BLOCK_B, BLOCK_W),
            order=(1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_ptr + pad_left,
            shape=(B, W_in),
            strides=(W_out, 1),
            offsets=(pid_b * BLOCK_B, pid_w * BLOCK_W),
            block_shape=(BLOCK_B, BLOCK_W),
            order=(1, 0),
        )
        vals = tl.load(in_block, boundary_check=(0, 1), padding_option="zero")
        tl.store(out_block, vals, boundary_check=(0, 1))

    # Phase 2: Edge gather — left+right reflection in same launch
    if edge_total > 0:
        n_left = B * left_count
        for block_id in tl.range(pid, tl.cdiv(edge_total, BLOCK_EDGE), GRID_DIM):
            offs = block_id * BLOCK_EDGE + tl.arange(0, BLOCK_EDGE)
            mask = offs < edge_total

            is_right = offs >= n_left

            # Left region
            left_offs = offs
            left_pid_b = left_offs // left_count
            left_w = left_offs % left_count
            left_iw = pad_left - left_w

            # Right region
            right_offs = offs - n_left
            right_pid_b = right_offs // right_count
            right_w = right_start + (right_offs % right_count)
            right_iw = 2 * W_in - 2 - right_w + pad_left

            pid_b = tl.where(is_right, right_pid_b, left_pid_b)
            w_out = tl.where(is_right, right_w, left_w)
            iw = tl.where(is_right, right_iw, left_iw)

            in_offs = pid_b * W_in + iw
            out_offs = pid_b * W_out + w_out

            vals = tl.load(in_ptr + in_offs, mask=mask, other=0)
            tl.store(out_ptr + out_offs, vals, mask=mask)


def _launch_reflection_pad1d(input, padding, out=None):
    if not isinstance(padding, (list, tuple)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of length 2: (pad_left, pad_right)"
        )
    pad_left, pad_right = int(padding[0]), int(padding[1])
    if pad_left < 0 or pad_right < 0:
        raise ValueError("padding values must be >= 0")
    if input.dim() < 1:
        raise ValueError("input must have at least 1 dimension")

    x = input.contiguous()
    W_in = int(x.shape[-1])
    if W_in <= 0:
        raise ValueError("last dimension (width) must be > 0")

    W_out = W_in + pad_left + pad_right
    leading_shape = x.shape[:-1]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    if out is None:
        out = torch.empty((*leading_shape, W_out), device=x.device, dtype=x.dtype)
    else:
        expected_shape = (*leading_shape, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, "
                f"expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    if pad_left == 0 and pad_right == 0:
        out.copy_(x)
        return out

    if W_in < 2:
        raise ValueError(
            "input width must be at least 2 " "for reflection padding when padding > 0"
        )
    if pad_left >= W_in or pad_right >= W_in:
        raise ValueError(
            "padding values must be less than " "the input width for reflection padding"
        )

    total_elems = B * W_out

    if total_elems < _TINY_THRESHOLD:
        # Tiny shape: aten flip+copy (lower dispatch overhead than Triton launch)
        if pad_left > 0:
            out.narrow(-1, 0, pad_left).copy_(x.narrow(-1, 1, pad_left).flip(-1))
        out.narrow(-1, pad_left, W_in).copy_(x)
        if pad_right > 0:
            out.narrow(-1, pad_left + W_in, pad_right).copy_(
                x.narrow(-1, W_in - pad_right - 1, pad_right).flip(-1)
            )
    else:
        # Medium+large shape: single fused two-phase kernel
        GRID_DIM = 24
        BLOCK_B = 16
        BLOCK_W = 1024
        BLOCK_EDGE = 8192
        grid = (GRID_DIM,)

        n_tiles_b = triton.cdiv(B, BLOCK_B)
        n_tiles_w = triton.cdiv(W_in, BLOCK_W)

        with torch_device_fn.device(x.device):
            fused_two_phase_kernel[grid](
                x,
                out,
                B,
                W_in,
                W_out,
                pad_left,
                pad_left,  # left_count
                pad_left + W_in,  # right_start
                pad_right,  # right_count
                B * (pad_left + pad_right),  # edge_total
                n_tiles_b,
                n_tiles_w,
                GRID_DIM=GRID_DIM,
                BLOCK_B=BLOCK_B,
                BLOCK_W=BLOCK_W,
                BLOCK_EDGE=BLOCK_EDGE,
            )

    return out


def reflection_pad1d(input, padding):
    logger.debug("GEMS_ENFLAME_GCU300 REFLECTION_PAD1D")
    return _launch_reflection_pad1d(input, padding, out=None)


def reflection_pad1d_out(input, padding, out):
    logger.debug("GEMS_ENFLAME_GCU300 REFLECTION_PAD1D_OUT")
    return _launch_reflection_pad1d(input, padding, out=out)
