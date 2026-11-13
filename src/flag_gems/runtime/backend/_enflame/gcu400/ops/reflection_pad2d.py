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
import math

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# GCU400 reflection_pad2d strategy (2026-11-11 perf rewrite):
# The old kernel walked the flat H_out*W_out output space with per-lane
# `// W_out` / `% W_out` decode plus abs/%/where reflection chains, launched
# as grid=(B, cdiv(H_out*W_out, 512)): the big bench shape paid ~135k
# programs on the ~52ns/program dispatch floor (measured latencies were
# dtype-independent), and every lane paid a gather-form load. The rewrite
# keeps one kernel launch and decomposes the op along output rows:
#   * phase 1 (bulk): every output row copies its W-wide interior segment
#     (source row = reflected h) as ONE flat `base + tl.arange` pair; U
#     consecutive rows are grouped so all U loads are in flight before the
#     U stores (hides the ~450ns per-op latency that serializes narrow
#     rows). Interior-band groups use an affine fast path (s_i = s0 + i*W);
#     tail / band-crossing / image-boundary groups take the per-row chain.
#   * phase 2 (pads): the left/right mirror lanes (pl+pr per row, a few
#     percent of the output) are copied by a per-lane gather loop, so bulk
#     programs never issue tiny ops.
# Top/bottom bands need no special casing: their interior middles ride the
# bulk row loop (reflected source row) and their mirror columns ride the
# pad gather, so corners are covered by the same two phases.
# Grid is capped at 48 (one program per SPC); rows/lanes are grid-strided.
# U (rows per group) dispatch mirrors the _pad_circular tiering: wide rows
# are bandwidth-bound (small U), narrow rows are latency-bound (large U).


@triton.jit
def _refl_row_src(r, W, Ho, H, plh):
    # scalar source row base (element offset) of output row r under h-reflection
    b = r // Ho
    h = r % Ho
    sh = h - plh
    sh = tl.where(sh < 0, -sh, sh)
    sh = tl.where(sh >= H, 2 * (H - 1) - sh, sh)
    return (b * H + sh) * W


@triton.jit
def reflection_pad2d_rowseg_kernel(
    x_ptr,
    o_ptr,
    R,
    W,
    Wo,
    plw,
    Ho,
    H,
    plh,
    BLOCK: tl.constexpr,
    GBLOCK: tl.constexpr,
    U: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    aM = plw  # interior segment starts at output column plw
    PW = Wo - W  # pl + pr mirror lanes per row
    offs = tl.arange(0, BLOCK)
    mb = offs < W
    GU = G * U
    for r0 in range(pid * U, R, GU):
        b0 = r0 // Ho
        h0 = r0 - b0 * Ho
        # affine fast path: whole group inside one image's interior band
        # (implies no image wrap; Ho = plh + H + pb >= plh + H)
        fast = (r0 + U <= R) & (h0 >= plh) & (h0 + U <= plh + H)
        if fast:
            s0 = (b0 * H + h0 - plh) * W
            d0 = r0 * Wo + aM
            if U == 4:
                v0 = tl.load(x_ptr + s0 + offs, mask=mb)
                v1 = tl.load(x_ptr + s0 + W + offs, mask=mb)
                v2 = tl.load(x_ptr + s0 + 2 * W + offs, mask=mb)
                v3 = tl.load(x_ptr + s0 + 3 * W + offs, mask=mb)
                tl.store(o_ptr + d0 + offs, v0, mask=mb)
                tl.store(o_ptr + d0 + Wo + offs, v1, mask=mb)
                tl.store(o_ptr + d0 + 2 * Wo + offs, v2, mask=mb)
                tl.store(o_ptr + d0 + 3 * Wo + offs, v3, mask=mb)
            elif U == 8:
                v0 = tl.load(x_ptr + s0 + offs, mask=mb)
                v1 = tl.load(x_ptr + s0 + W + offs, mask=mb)
                v2 = tl.load(x_ptr + s0 + 2 * W + offs, mask=mb)
                v3 = tl.load(x_ptr + s0 + 3 * W + offs, mask=mb)
                v4 = tl.load(x_ptr + s0 + 4 * W + offs, mask=mb)
                v5 = tl.load(x_ptr + s0 + 5 * W + offs, mask=mb)
                v6 = tl.load(x_ptr + s0 + 6 * W + offs, mask=mb)
                v7 = tl.load(x_ptr + s0 + 7 * W + offs, mask=mb)
                tl.store(o_ptr + d0 + offs, v0, mask=mb)
                tl.store(o_ptr + d0 + Wo + offs, v1, mask=mb)
                tl.store(o_ptr + d0 + 2 * Wo + offs, v2, mask=mb)
                tl.store(o_ptr + d0 + 3 * Wo + offs, v3, mask=mb)
                tl.store(o_ptr + d0 + 4 * Wo + offs, v4, mask=mb)
                tl.store(o_ptr + d0 + 5 * Wo + offs, v5, mask=mb)
                tl.store(o_ptr + d0 + 6 * Wo + offs, v6, mask=mb)
                tl.store(o_ptr + d0 + 7 * Wo + offs, v7, mask=mb)
            else:
                v0 = tl.load(x_ptr + s0 + offs, mask=mb)
                v1 = tl.load(x_ptr + s0 + W + offs, mask=mb)
                v2 = tl.load(x_ptr + s0 + 2 * W + offs, mask=mb)
                v3 = tl.load(x_ptr + s0 + 3 * W + offs, mask=mb)
                v4 = tl.load(x_ptr + s0 + 4 * W + offs, mask=mb)
                v5 = tl.load(x_ptr + s0 + 5 * W + offs, mask=mb)
                v6 = tl.load(x_ptr + s0 + 6 * W + offs, mask=mb)
                v7 = tl.load(x_ptr + s0 + 7 * W + offs, mask=mb)
                v8 = tl.load(x_ptr + s0 + 8 * W + offs, mask=mb)
                v9 = tl.load(x_ptr + s0 + 9 * W + offs, mask=mb)
                v10 = tl.load(x_ptr + s0 + 10 * W + offs, mask=mb)
                v11 = tl.load(x_ptr + s0 + 11 * W + offs, mask=mb)
                v12 = tl.load(x_ptr + s0 + 12 * W + offs, mask=mb)
                v13 = tl.load(x_ptr + s0 + 13 * W + offs, mask=mb)
                v14 = tl.load(x_ptr + s0 + 14 * W + offs, mask=mb)
                v15 = tl.load(x_ptr + s0 + 15 * W + offs, mask=mb)
                v16 = tl.load(x_ptr + s0 + 16 * W + offs, mask=mb)
                v17 = tl.load(x_ptr + s0 + 17 * W + offs, mask=mb)
                v18 = tl.load(x_ptr + s0 + 18 * W + offs, mask=mb)
                v19 = tl.load(x_ptr + s0 + 19 * W + offs, mask=mb)
                tl.store(o_ptr + d0 + offs, v0, mask=mb)
                tl.store(o_ptr + d0 + Wo + offs, v1, mask=mb)
                tl.store(o_ptr + d0 + 2 * Wo + offs, v2, mask=mb)
                tl.store(o_ptr + d0 + 3 * Wo + offs, v3, mask=mb)
                tl.store(o_ptr + d0 + 4 * Wo + offs, v4, mask=mb)
                tl.store(o_ptr + d0 + 5 * Wo + offs, v5, mask=mb)
                tl.store(o_ptr + d0 + 6 * Wo + offs, v6, mask=mb)
                tl.store(o_ptr + d0 + 7 * Wo + offs, v7, mask=mb)
                tl.store(o_ptr + d0 + 8 * Wo + offs, v8, mask=mb)
                tl.store(o_ptr + d0 + 9 * Wo + offs, v9, mask=mb)
                tl.store(o_ptr + d0 + 10 * Wo + offs, v10, mask=mb)
                tl.store(o_ptr + d0 + 11 * Wo + offs, v11, mask=mb)
                tl.store(o_ptr + d0 + 12 * Wo + offs, v12, mask=mb)
                tl.store(o_ptr + d0 + 13 * Wo + offs, v13, mask=mb)
                tl.store(o_ptr + d0 + 14 * Wo + offs, v14, mask=mb)
                tl.store(o_ptr + d0 + 15 * Wo + offs, v15, mask=mb)
                tl.store(o_ptr + d0 + 16 * Wo + offs, v16, mask=mb)
                tl.store(o_ptr + d0 + 17 * Wo + offs, v17, mask=mb)
                tl.store(o_ptr + d0 + 18 * Wo + offs, v18, mask=mb)
                tl.store(o_ptr + d0 + 19 * Wo + offs, v19, mask=mb)
        elif U == 4:
            r_0 = tl.minimum(r0, R - 1)
            r_1 = tl.minimum(r0 + 1, R - 1)
            r_2 = tl.minimum(r0 + 2, R - 1)
            r_3 = tl.minimum(r0 + 3, R - 1)
            s_0 = _refl_row_src(r_0, W, Ho, H, plh)
            s_1 = _refl_row_src(r_1, W, Ho, H, plh)
            s_2 = _refl_row_src(r_2, W, Ho, H, plh)
            s_3 = _refl_row_src(r_3, W, Ho, H, plh)
            v0 = tl.load(x_ptr + s_0 + offs, mask=mb)
            v1 = tl.load(x_ptr + s_1 + offs, mask=mb)
            v2 = tl.load(x_ptr + s_2 + offs, mask=mb)
            v3 = tl.load(x_ptr + s_3 + offs, mask=mb)
            tl.store(o_ptr + r_0 * Wo + aM + offs, v0, mask=mb)
            tl.store(o_ptr + r_1 * Wo + aM + offs, v1, mask=mb)
            tl.store(o_ptr + r_2 * Wo + aM + offs, v2, mask=mb)
            tl.store(o_ptr + r_3 * Wo + aM + offs, v3, mask=mb)
        elif U == 8:
            r_0 = tl.minimum(r0, R - 1)
            r_1 = tl.minimum(r0 + 1, R - 1)
            r_2 = tl.minimum(r0 + 2, R - 1)
            r_3 = tl.minimum(r0 + 3, R - 1)
            r_4 = tl.minimum(r0 + 4, R - 1)
            r_5 = tl.minimum(r0 + 5, R - 1)
            r_6 = tl.minimum(r0 + 6, R - 1)
            r_7 = tl.minimum(r0 + 7, R - 1)
            s_0 = _refl_row_src(r_0, W, Ho, H, plh)
            s_1 = _refl_row_src(r_1, W, Ho, H, plh)
            s_2 = _refl_row_src(r_2, W, Ho, H, plh)
            s_3 = _refl_row_src(r_3, W, Ho, H, plh)
            s_4 = _refl_row_src(r_4, W, Ho, H, plh)
            s_5 = _refl_row_src(r_5, W, Ho, H, plh)
            s_6 = _refl_row_src(r_6, W, Ho, H, plh)
            s_7 = _refl_row_src(r_7, W, Ho, H, plh)
            v0 = tl.load(x_ptr + s_0 + offs, mask=mb)
            v1 = tl.load(x_ptr + s_1 + offs, mask=mb)
            v2 = tl.load(x_ptr + s_2 + offs, mask=mb)
            v3 = tl.load(x_ptr + s_3 + offs, mask=mb)
            v4 = tl.load(x_ptr + s_4 + offs, mask=mb)
            v5 = tl.load(x_ptr + s_5 + offs, mask=mb)
            v6 = tl.load(x_ptr + s_6 + offs, mask=mb)
            v7 = tl.load(x_ptr + s_7 + offs, mask=mb)
            tl.store(o_ptr + r_0 * Wo + aM + offs, v0, mask=mb)
            tl.store(o_ptr + r_1 * Wo + aM + offs, v1, mask=mb)
            tl.store(o_ptr + r_2 * Wo + aM + offs, v2, mask=mb)
            tl.store(o_ptr + r_3 * Wo + aM + offs, v3, mask=mb)
            tl.store(o_ptr + r_4 * Wo + aM + offs, v4, mask=mb)
            tl.store(o_ptr + r_5 * Wo + aM + offs, v5, mask=mb)
            tl.store(o_ptr + r_6 * Wo + aM + offs, v6, mask=mb)
            tl.store(o_ptr + r_7 * Wo + aM + offs, v7, mask=mb)
        else:
            r_0 = tl.minimum(r0, R - 1)
            r_1 = tl.minimum(r0 + 1, R - 1)
            r_2 = tl.minimum(r0 + 2, R - 1)
            r_3 = tl.minimum(r0 + 3, R - 1)
            r_4 = tl.minimum(r0 + 4, R - 1)
            r_5 = tl.minimum(r0 + 5, R - 1)
            r_6 = tl.minimum(r0 + 6, R - 1)
            r_7 = tl.minimum(r0 + 7, R - 1)
            r_8 = tl.minimum(r0 + 8, R - 1)
            r_9 = tl.minimum(r0 + 9, R - 1)
            r_10 = tl.minimum(r0 + 10, R - 1)
            r_11 = tl.minimum(r0 + 11, R - 1)
            r_12 = tl.minimum(r0 + 12, R - 1)
            r_13 = tl.minimum(r0 + 13, R - 1)
            r_14 = tl.minimum(r0 + 14, R - 1)
            r_15 = tl.minimum(r0 + 15, R - 1)
            r_16 = tl.minimum(r0 + 16, R - 1)
            r_17 = tl.minimum(r0 + 17, R - 1)
            r_18 = tl.minimum(r0 + 18, R - 1)
            r_19 = tl.minimum(r0 + 19, R - 1)
            s_0 = _refl_row_src(r_0, W, Ho, H, plh)
            s_1 = _refl_row_src(r_1, W, Ho, H, plh)
            s_2 = _refl_row_src(r_2, W, Ho, H, plh)
            s_3 = _refl_row_src(r_3, W, Ho, H, plh)
            s_4 = _refl_row_src(r_4, W, Ho, H, plh)
            s_5 = _refl_row_src(r_5, W, Ho, H, plh)
            s_6 = _refl_row_src(r_6, W, Ho, H, plh)
            s_7 = _refl_row_src(r_7, W, Ho, H, plh)
            s_8 = _refl_row_src(r_8, W, Ho, H, plh)
            s_9 = _refl_row_src(r_9, W, Ho, H, plh)
            s_10 = _refl_row_src(r_10, W, Ho, H, plh)
            s_11 = _refl_row_src(r_11, W, Ho, H, plh)
            s_12 = _refl_row_src(r_12, W, Ho, H, plh)
            s_13 = _refl_row_src(r_13, W, Ho, H, plh)
            s_14 = _refl_row_src(r_14, W, Ho, H, plh)
            s_15 = _refl_row_src(r_15, W, Ho, H, plh)
            s_16 = _refl_row_src(r_16, W, Ho, H, plh)
            s_17 = _refl_row_src(r_17, W, Ho, H, plh)
            s_18 = _refl_row_src(r_18, W, Ho, H, plh)
            s_19 = _refl_row_src(r_19, W, Ho, H, plh)
            v0 = tl.load(x_ptr + s_0 + offs, mask=mb)
            v1 = tl.load(x_ptr + s_1 + offs, mask=mb)
            v2 = tl.load(x_ptr + s_2 + offs, mask=mb)
            v3 = tl.load(x_ptr + s_3 + offs, mask=mb)
            v4 = tl.load(x_ptr + s_4 + offs, mask=mb)
            v5 = tl.load(x_ptr + s_5 + offs, mask=mb)
            v6 = tl.load(x_ptr + s_6 + offs, mask=mb)
            v7 = tl.load(x_ptr + s_7 + offs, mask=mb)
            v8 = tl.load(x_ptr + s_8 + offs, mask=mb)
            v9 = tl.load(x_ptr + s_9 + offs, mask=mb)
            v10 = tl.load(x_ptr + s_10 + offs, mask=mb)
            v11 = tl.load(x_ptr + s_11 + offs, mask=mb)
            v12 = tl.load(x_ptr + s_12 + offs, mask=mb)
            v13 = tl.load(x_ptr + s_13 + offs, mask=mb)
            v14 = tl.load(x_ptr + s_14 + offs, mask=mb)
            v15 = tl.load(x_ptr + s_15 + offs, mask=mb)
            v16 = tl.load(x_ptr + s_16 + offs, mask=mb)
            v17 = tl.load(x_ptr + s_17 + offs, mask=mb)
            v18 = tl.load(x_ptr + s_18 + offs, mask=mb)
            v19 = tl.load(x_ptr + s_19 + offs, mask=mb)
            tl.store(o_ptr + r_0 * Wo + aM + offs, v0, mask=mb)
            tl.store(o_ptr + r_1 * Wo + aM + offs, v1, mask=mb)
            tl.store(o_ptr + r_2 * Wo + aM + offs, v2, mask=mb)
            tl.store(o_ptr + r_3 * Wo + aM + offs, v3, mask=mb)
            tl.store(o_ptr + r_4 * Wo + aM + offs, v4, mask=mb)
            tl.store(o_ptr + r_5 * Wo + aM + offs, v5, mask=mb)
            tl.store(o_ptr + r_6 * Wo + aM + offs, v6, mask=mb)
            tl.store(o_ptr + r_7 * Wo + aM + offs, v7, mask=mb)
            tl.store(o_ptr + r_8 * Wo + aM + offs, v8, mask=mb)
            tl.store(o_ptr + r_9 * Wo + aM + offs, v9, mask=mb)
            tl.store(o_ptr + r_10 * Wo + aM + offs, v10, mask=mb)
            tl.store(o_ptr + r_11 * Wo + aM + offs, v11, mask=mb)
            tl.store(o_ptr + r_12 * Wo + aM + offs, v12, mask=mb)
            tl.store(o_ptr + r_13 * Wo + aM + offs, v13, mask=mb)
            tl.store(o_ptr + r_14 * Wo + aM + offs, v14, mask=mb)
            tl.store(o_ptr + r_15 * Wo + aM + offs, v15, mask=mb)
            tl.store(o_ptr + r_16 * Wo + aM + offs, v16, mask=mb)
            tl.store(o_ptr + r_17 * Wo + aM + offs, v17, mask=mb)
            tl.store(o_ptr + r_18 * Wo + aM + offs, v18, mask=mb)
            tl.store(o_ptr + r_19 * Wo + aM + offs, v19, mask=mb)
    # ---- phase 2: left/right mirror lanes via per-lane gather ----
    total = R * PW
    if PW > 0:
        goffs = tl.arange(0, GBLOCK)
        for t0 in range(pid * GBLOCK, total, G * GBLOCK):
            t = t0 + goffs
            m = t < total
            row = t // PW
            rowc = tl.minimum(row, R - 1)
            j = t - row * PW
            is_left = j < plw
            # left lane j: out col j, src col plw - j (mirror around col 0)
            # right lane j: out col W + j, src col W - 2 - (j - plw)
            dj = tl.where(is_left, j, W + j)
            sj = tl.where(is_left, plw - j, W - 2 - j + plw)
            srow = _refl_row_src(rowc, W, Ho, H, plh)
            tl.store(o_ptr + rowc * Wo + dj, tl.load(x_ptr + srow + sj, mask=m), mask=m)


@triton.jit
def reflection_pad2d_wide_kernel(
    x_ptr,
    o_ptr,
    R,
    W,
    Wo,
    plw,
    Ho,
    H,
    plh,
    BLOCK: tl.constexpr,
    FPAD: tl.constexpr,
):
    # general path for very wide rows (W > 8192): per-row chunked interior
    # copy plus chunked mirror copies; correctness-first, off the hot path.
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    prw = Wo - W - plw
    CB = (W + BLOCK - 1) // BLOCK
    CL = (plw + FPAD - 1) // FPAD
    CR = (prw + FPAD - 1) // FPAD
    offs_b = tl.arange(0, BLOCK)
    offs_f = tl.arange(0, FPAD)
    for r in range(pid, R, G):
        srow = _refl_row_src(r, W, Ho, H, plh)
        drow = r * Wo
        for c in range(0, CB):
            ob = c * BLOCK + offs_b
            m = ob < W
            tl.store(
                o_ptr + drow + plw + ob,
                tl.load(x_ptr + srow + ob, mask=m),
                mask=m,
            )
        for c in range(0, CL):
            ol = c * FPAD + offs_f
            ml = ol < plw
            sl = tl.where(ml, plw - ol, 0)
            tl.store(o_ptr + drow + ol, tl.load(x_ptr + srow + sl, mask=ml), mask=ml)
        for c in range(0, CR):
            orr = c * FPAD + offs_f
            mr = orr < prw
            sr = tl.where(mr, W - 2 - orr, 0)
            tl.store(
                o_ptr + drow + W + plw + orr,
                tl.load(x_ptr + srow + sr, mask=mr),
                mask=mr,
            )


def _next_pow2(n, cap):
    p = 1
    while p < n and p < cap:
        p *= 2
    return p


def launch_reflection_pad2d(input: torch.Tensor, padding, out: torch.Tensor = None):
    # Validate padding format
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    # Validate padding values
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be >= 0")

    # Validate input
    if input.dim() < 3:
        raise ValueError("input must have at least 3 dimensions")
    if input.device.type != flag_gems.device:
        raise ValueError(f"input must be a {flag_gems.device} tensor")

    x = input.contiguous()
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    # Validate reflection padding constraints
    if H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 for reflection padding when padding > 0"
        )
    if H_in <= 0 or W_in <= 0:
        raise ValueError("spatial dimensions must be > 0")
    if pad_left >= W_in or pad_right >= W_in or pad_top >= H_in or pad_bottom >= H_in:
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Handle output tensor
    if out is None:
        out = torch.empty(
            (*leading_shape, H_out, W_out), device=x.device, dtype=x.dtype
        )
    else:
        if out.device.type != flag_gems.device:
            raise ValueError(f"out must be a {flag_gems.device} tensor")
        expected_shape = (*leading_shape, H_out, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    R = B * H_out
    PW = W_out - W_in
    pad_right_w = PW - pad_left

    def _launch():
        if W_in > 8192:
            # very wide rows: chunked general path
            FPAD = max(16, _next_pow2(max(pad_left, pad_right_w, 1), 1024))
            grid = (min(48, max(1, R)),)
            reflection_pad2d_wide_kernel[grid](
                x,
                out,
                R,
                W_in,
                W_out,
                pad_left,
                H_out,
                H_in,
                pad_top,
                BLOCK=8192,
                FPAD=FPAD,
                num_warps=4,
            )
        else:
            # rows per group: wide rows are bandwidth-bound (small groups),
            # narrow rows are latency-bound (large groups hide the ~450ns
            # per-op latency). U=20 was measured best for W in [64, 768);
            # U=32 measured worse on every shape (register/issue cliff).
            # Small R clamp: when fewer groups than programs would result,
            # prefer more programs with smaller groups (measured on the
            # (3,33,33)/(2,4,32,64) bench shapes).
            if W_in >= 2048:
                U = 4
            elif W_in >= 768 or W_in < 64:
                U = 8
            else:
                U = 20
            n_min = (R + 47) // 48
            if n_min <= 4:
                U_r = 4
            elif n_min <= 8:
                U_r = 8
            else:
                U_r = 20
            U = min(U, U_r)
            nw = 2 if U == 20 else 4
            BLOCK = _next_pow2(W_in, 8192)
            total = R * PW
            grid = (min(48, max(1, (R + U - 1) // U + (total + 1023) // 1024)),)
            reflection_pad2d_rowseg_kernel[grid](
                x,
                out,
                R,
                W_in,
                W_out,
                pad_left,
                H_out,
                H_in,
                pad_top,
                BLOCK=BLOCK,
                GBLOCK=1024,
                U=U,
                num_warps=nw,
            )

    # cold compiles hit the documented node-level toolchain flakiness
    # (P02 family): a fresh-compile subprocess can die before producing a
    # cache entry; retrying the launch re-enters compilation and succeeds
    # (atan2_/sinh/_pad_circular precedent). Zero cost on the warm path.
    for attempt in range(3):
        try:
            with torch_device_fn.device(x.device):
                _launch()
            break
        except Exception:
            if attempt == 2:
                raise
    return out


def reflection_pad2d(input: torch.Tensor, padding):
    logger.debug("GEMS_ENFLAME REFLECTION_PAD2D")
    return launch_reflection_pad2d(input, padding, out=None)


def reflection_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME REFLECTION_PAD2D_OUT")
    return launch_reflection_pad2d(input, padding, out=out)
