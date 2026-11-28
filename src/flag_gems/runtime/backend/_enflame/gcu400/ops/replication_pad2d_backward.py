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

"""replication_pad2d backward (Enflame GCU400): single launch, no atomics, row-grid striding implementation.

2026-11-15 fix + optimization
=====================
1) **grid.x overflow crash (functional defect)**
   The old implementation had two paths: with interior_elements <= 16384 a "one program per row" fused kernel,
   otherwise a fallback of "interior copy + edge atomic add" two kernels. The fallback grid is
   `nc_total * H * cdiv(W, INTERIOR_BLOCK)` / `cdiv(interior_elements, BLOCK)`,
   which needs 131072 for (4,64,256,512) and more for (8,128,1024,1024), exceeding the hardware grid.x <= 65535,
   directly raising `OutOfResources: out of resource: grid.x` (all 18 cases of 4x64x256x512 in tests
   failed). The new implementation puts the "input row" loop inside the kernel as grid-stride
   (`for row in range(pid, ROWS, G)`), so grid depends only on row count and tuning caps and no shape
   can overflow; the two-kernel fallback is also removed (its accuracy defect is point 2).
2) **fp16/bf16 edge accuracy**
   The old edge kernel used `tl.atomic_add` to accumulate directly into low-precision output, rounding once per pad
   element
   (comparing random grad_output with a CPU float64 reference, 0.1% of boundary elements differ by
   0.047). The new implementation accumulates all output elements mapping to one input element in fp32 in registers,
   casts once at the end, and needs no atomics at all.
3) **ph dimension changed to sequential accumulation (main performance source)**
   A `[BLOCK_PH, BLOCK_W]` 2D tile followed by `tl.sum(axis=0)` becomes a cross-lane reduction on GCU,
   the biggest cost of this kernel; switching to `tl.static_range(BLOCK_PH)` sequential accumulation into a 1D register
   vector raised the same case group's geomean from 17.19x to 37.59x (+2.2x, see report). Column direction is blocked by
   `ELEM_CAP`, so extra-wide rows (W > BLOCK_W) do not blow up registers.

Mapping semantics (replication backward): input row (nc, ih) receives output rows in the closed interval
[oh_lo, oh_hi] = [ih == 0 ? 0 : PT + ih, ih == H-1 ? OH-1 : PT+ih]; in the column direction left pad
columns fold to iw = 0 and right pad columns fold to iw = W-1. Each grad_output element is read once,
so read/write traffic is OH*OW reads + H*W writes (optimal).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# ===== Tuning constants (calibrated by interleaved A/B measurements on 2026-11-17, do_bench median w1000/r100) =====
# grid = min(work count, SPC * GRID_MULT, MAX_GRID): GRID_MULT=1 (48 programs) is best,
# 4x/16x are -0.4%/-3.6% respectively (large shapes); no gain for small shapes.
GRID_MULT = 1
SPC = 48
MAX_GRID = 65535
# 2D tile lanes budget: 8192 is best; 16384/32768 are -26%/-80% respectively (DSM/register pressure).
# Also the upper bound of the masked tile safe range (nanmean precedent: non-EVEN masked load lanes<=8192).
LANE_BUDGET = 8192
# Upper bound on BLOCK_R rows; the actual value takes the largest power of 2 dividing H and <= H-1 (to avoid crossing
# segments + at most
# 2 edge rows per tile).
MAX_BLOCK_R = 32
# num_warps: 1 is best (-3% at 2); nw>=4 triggers silent wrong results of the P10 family, so it is disabled.
WARPS = 1
# ============================================================================


@triton.jit
def _rp2d_bwd_row(
    grad_output_ptr,
    grad_input_ptr,
    row,
    H,
    W,
    OH,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    GO_NC: tl.constexpr,
    GO_ROW: tl.constexpr,
    iw0,
    pw,
    BLOCK_W: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    NCHUNK: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Generic per-row body (the 2026-11-15 logic): shared by edge rows, non-EVEN shapes and tails."""
    nc = row // H
    ih = row - nc * H
    oh_lo = tl.where(ih == 0, 0, PT + ih)
    oh_hi = tl.where(ih == H - 1, OH - 1, PT + ih)
    go_nc = grad_output_ptr + nc * GO_NC
    gi_row = grad_input_ptr + row * W
    for chunk in tl.static_range(NCHUNK):
        iw = chunk * BLOCK_W + iw0
        iwv = iw < W
        total = tl.zeros([BLOCK_W], dtype=tl.float32)
        for p in tl.static_range(BLOCK_PH):
            oh = oh_lo + p
            if oh <= oh_hi:
                total += tl.load(
                    go_nc + oh * GO_ROW + (PL + iw), mask=iwv, other=0.0
                ).to(tl.float32)
        if PL > 0:
            if chunk == 0:
                left_sum = 0.0
                for p in tl.static_range(BLOCK_PH):
                    oh = oh_lo + p
                    if oh <= oh_hi:
                        left_sum += tl.sum(
                            tl.load(
                                go_nc + oh * GO_ROW + pw, mask=pw < PL, other=0.0
                            ).to(tl.float32)
                        )
                total += tl.where(iw0 == 0, left_sum, 0.0)
        if PR > 0:
            if chunk == NCHUNK - 1:
                right_sum = 0.0
                for p in tl.static_range(BLOCK_PH):
                    oh = oh_lo + p
                    if oh <= oh_hi:
                        right_sum += tl.sum(
                            tl.load(
                                go_nc + oh * GO_ROW + (PL + W + pw),
                                mask=pw < PR,
                                other=0.0,
                            ).to(tl.float32)
                        )
                total += tl.where(iw == W - 1, right_sum, 0.0)
        tl.store(gi_row + iw, total.to(OUT_DTYPE), mask=iwv)


@triton.jit
def replication_pad2d_backward_tile_kernel(
    grad_output_ptr,
    grad_input_ptr,
    NTILES,
    ROWS,
    H,
    W,
    OH,
    OW,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
    GO_NC: tl.constexpr,
    GO_ROW: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    NCHUNK: tl.constexpr,
    EVEN_W: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    iw0 = tl.arange(0, BLOCK_W)
    pw = tl.arange(0, BLOCK_PAD)
    rr = tl.arange(0, BLOCK_R)

    if EVEN_W:
        # Loop A: fast path - unmasked 2D row tile (interior rows injective + pad column folding).
        # Edge rows read placeholder values at clamped addresses here, which are later overwritten by loop B in the same
        # program.
        for tile in range(pid, NTILES, G):
            row0 = tile * BLOCK_R
            ih0 = row0 % H
            ih_vec = ih0 + rr
            ih_safe = tl.minimum(tl.maximum(ih_vec, 1), tl.maximum(H - 2, 0))
            nc_vec = row0 // H
            oh_vec = PT + ih_safe
            go_row = grad_output_ptr + nc_vec * GO_NC + oh_vec * GO_ROW

            left_v = tl.zeros([BLOCK_R], dtype=tl.float32)
            if PL > 0:
                for p in tl.static_range(BLOCK_PAD):
                    col = tl.minimum(p, OW - 1)
                    v = tl.load(go_row + col).to(tl.float32)
                    left_v += tl.where(p < PL, v, 0.0)
            right_v = tl.zeros([BLOCK_R], dtype=tl.float32)
            if PR > 0:
                for p in tl.static_range(BLOCK_PAD):
                    col = tl.minimum(PL + W + p, OW - 1)
                    v = tl.load(go_row + col).to(tl.float32)
                    right_v += tl.where(p < PR, v, 0.0)

            inter = tl.load(go_row[:, None] + (PL + iw0)[None, :])
            total = inter.to(tl.float32)
            if PL > 0:
                total += tl.where(iw0[None, :] == 0, left_v[:, None], 0.0)
            if PR > 0:
                total += tl.where(iw0[None, :] == W - 1, right_v[:, None], 0.0)
            tl.store(
                grad_input_ptr + (row0 + rr)[:, None] * W + iw0[None, :],
                total.to(OUT_DTYPE),
            )

        # Loop B: recomputation of edge rows within this program's tile (separate from loop A:
        # putting the helper in the same loop as the 2D tile body triggers an in-process SIGABRT, see module docstring).
        for tile in range(pid, NTILES, G):
            row0 = tile * BLOCK_R
            ih0 = row0 % H
            if ih0 == 0:
                _rp2d_bwd_row(
                    grad_output_ptr,
                    grad_input_ptr,
                    row0,
                    H,
                    W,
                    OH,
                    PL,
                    PR,
                    PT,
                    GO_NC,
                    GO_ROW,
                    iw0,
                    pw,
                    BLOCK_W,
                    BLOCK_PH,
                    NCHUNK,
                    OUT_DTYPE,
                )
            if (H - 1 - ih0) < BLOCK_R:
                _rp2d_bwd_row(
                    grad_output_ptr,
                    grad_input_ptr,
                    row0 + (H - 1 - ih0),
                    H,
                    W,
                    OH,
                    PL,
                    PR,
                    PT,
                    GO_NC,
                    GO_ROW,
                    iw0,
                    pw,
                    BLOCK_W,
                    BLOCK_PH,
                    NCHUNK,
                    OUT_DTYPE,
                )

        # Loop C: fallback for tail rows (when EVEN_W, BLOCK_R | H, rows are always divisible, so effectively empty).
        for row in range(NTILES * BLOCK_R + pid, ROWS, G):
            _rp2d_bwd_row(
                grad_output_ptr,
                grad_input_ptr,
                row,
                H,
                W,
                OH,
                PL,
                PR,
                PT,
                GO_NC,
                GO_ROW,
                iw0,
                pw,
                BLOCK_W,
                BLOCK_PH,
                NCHUNK,
                OUT_DTYPE,
            )
    else:
        # General path: the same row loop as the 2026-11-15 row-by-row implementation.
        for row in range(pid, ROWS, G):
            _rp2d_bwd_row(
                grad_output_ptr,
                grad_input_ptr,
                row,
                H,
                W,
                OH,
                PL,
                PR,
                PT,
                GO_NC,
                GO_ROW,
                iw0,
                pw,
                BLOCK_W,
                BLOCK_PH,
                NCHUNK,
                OUT_DTYPE,
            )


def _dtype_constexpr(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float64:
        return tl.float64
    return tl.float32


def _next_pow2(n):
    value = 1
    while value < n:
        value *= 2
    return value


def _replication_pad2d_backward_impl(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, out: torch.Tensor = None
) -> torch.Tensor:
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 4:
            raise ValueError(
                "padding must be a sequence of 4 integers: "
                "(pad_left, pad_right, pad_top, pad_bottom)"
            )
        pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    else:
        raise TypeError(f"Unexpected padding type: {type(padding)}")

    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("Padding values must be non-negative")

    is_3d = self.ndim == 3
    if is_3d:
        C, H, W = self.shape
        N = 1
        grad_output_4d = grad_output.view(
            1, C, grad_output.shape[-2], grad_output.shape[-1]
        )
    elif self.ndim == 4:
        N, C, H, W = self.shape
        grad_output_4d = grad_output.contiguous()
    else:
        raise ValueError(
            "replication_pad2d_backward expects a 3D (C, H, W) or 4D (N, C, H, W) input"
        )

    OH = H + pad_top + pad_bottom
    OW = W + pad_left + pad_right

    expected_grad_shape = (N, C, OH, OW)
    if tuple(grad_output_4d.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output shape {tuple(grad_output_4d.shape)} does not match "
            f"expected shape {expected_grad_shape}"
        )

    # Without padding, backward is an identity copy
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        result = grad_output_4d.contiguous().to(self.dtype)
        if is_3d:
            result = result.view(C, H, W)
        if out is not None:
            out.copy_(result)
            return out
        return result

    rows = N * C * H
    if rows == 0:
        if out is not None:
            return out
        return torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)

    # Kernel writes the contiguous buffer directly with constant strides; non-contiguous out is computed into a
    # contiguous buffer then copied back
    write_back = None
    if out is None:
        target = torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)
    elif out.is_contiguous():
        target = out
    else:
        target = torch.empty((N, C, H, W), device=self.device, dtype=self.dtype)
        write_back = out

    block_w = min(_next_pow2(W), LANE_BUDGET)
    lanes = LANE_BUDGET // block_w
    # BLOCK_R must divide H: a tile must not cross (nc) segments; and <= H-1: at most 2 edge rows per tile
    block_r = 1
    while (block_r * 2 <= min(MAX_BLOCK_R, H - 1, lanes)) and (H % (block_r * 2) == 0):
        block_r *= 2
    nchunk = triton.cdiv(W, block_w)
    block_pad = _next_pow2(max(pad_left, pad_right, 2))
    block_ph = _next_pow2(1 + pad_top + pad_bottom)
    # The fast path is limited to a single chunk, power-of-2 width, and the presence of interior rows (H >= 3); when H
    # <= 2 all rows are
    # edge rows, and the general path is optimal.
    even_w = W == block_w and nchunk == 1 and H >= 3
    if even_w:
        ntiles = rows // block_r
        grid = min(max(ntiles, 1), SPC * GRID_MULT, MAX_GRID)
    else:
        ntiles = 0
        grid = min(rows, SPC * GRID_MULT, MAX_GRID)

    with torch_device_fn.device(self.device):
        replication_pad2d_backward_tile_kernel[(grid,)](
            grad_output_4d,
            target,
            ntiles,
            rows,
            H,
            W,
            OH,
            OW,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            OH * OW,
            OW,
            BLOCK_R=block_r,
            BLOCK_W=block_w,
            BLOCK_PAD=block_pad,
            BLOCK_PH=block_ph,
            NCHUNK=nchunk,
            EVEN_W=even_w,
            OUT_DTYPE=_dtype_constexpr(self.dtype),
            num_warps=WARPS,
        )

    result = target.view(C, H, W) if is_3d else target
    if write_back is not None:
        write_back.copy_(result)
        return write_back
    return result


def replication_pad2d_backward(grad_output, self, padding):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding)


def replication_pad2d_backward_grad_input(grad_output, self, padding, *, grad_input):
    logger.debug("GEMS_ENFLAME REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
